"""A PyG/ProG port of the local GraphMAE2 implementation.

The upstream project is DGL based.  This port preserves its pretraining
signals while using ProG's GNN: masked feature reconstruction is repeated
after decoder-side remasking and a BYOL-style EMA encoder supplies the latent
target.  Only ``self.gnn`` is exported, making the resulting checkpoint
loadable by every existing ProG downstream task.
"""

from __future__ import annotations

import copy
import time

import torch
from torch import nn

from prompt_graph.model import GNN
from prompt_graph.utils import get_logger

from ._masked_utils import (
    pretrain_metadata,
    resolve_node_data,
    safe_index_subset,
    save_encoder_checkpoint,
    validate_rate,
)
from .base import PreTrain
from .GraphMAE import sce_loss

logger = get_logger(__name__)


class GraphMAE2(PreTrain):
    """GraphMAE2-style full-batch node pretraining for ProG backbones."""

    def __init__(
        self,
        *args,
        data=None,
        mask_rate=0.5,
        remask_rate=0.5,
        num_remasking=3,
        lam=1.0,
        momentum=0.996,
        replace_rate=0.0,
        loss_fn="sce",
        alpha_l=2.0,
        output_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data, self.input_dim, self.output_dim = resolve_node_data(
            self.dataset_name, data, self.device
        )
        self.mask_rate = validate_rate("mask_rate", mask_rate, allow_zero=False)
        self.remask_rate = validate_rate("remask_rate", remask_rate)
        if int(num_remasking) < 1:
            raise ValueError("num_remasking must be >= 1")
        self.num_remasking = int(num_remasking)
        self.lam = float(lam)
        self.momentum = float(momentum)
        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        self.replace_rate = validate_rate("replace_rate", replace_rate)
        if loss_fn not in ("sce", "mse"):
            raise ValueError("loss_fn must be 'sce' or 'mse'")
        self.loss_fn = loss_fn
        self.alpha_l = float(alpha_l)
        self.output_path = output_path

        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.decoder = GNN(
            input_dim=self.hid_dim,
            hid_dim=self.hid_dim,
            out_dim=self.input_dim,
            num_layer=self.num_layer,
            conv_type=self.gnn_type,
        ).to(self.device)
        self.encoder_to_decoder = nn.Linear(self.hid_dim, self.hid_dim, bias=False).to(self.device)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.input_dim, device=self.device))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.hid_dim, device=self.device))

        self.projector = nn.Sequential(
            nn.Linear(self.hid_dim, 256),
            nn.PReLU(),
            nn.Linear(256, self.hid_dim),
        ).to(self.device)
        self.projector_ema = copy.deepcopy(self.projector)
        self.predictor = nn.Sequential(nn.PReLU(), nn.Linear(self.hid_dim, self.hid_dim)).to(
            self.device
        )
        self.encoder_ema = copy.deepcopy(self.gnn)
        for module in (self.encoder_ema, self.projector_ema):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            [
                *self.gnn.parameters(),
                *self.decoder.parameters(),
                *self.encoder_to_decoder.parameters(),
                *self.projector.parameters(),
                *self.predictor.parameters(),
                self.enc_mask_token,
                self.dec_mask_token,
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _mask_features(self):
        num_nodes = self.data.num_nodes
        mask_nodes = safe_index_subset(num_nodes, self.mask_rate, self.device)
        keep_nodes = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
        keep_nodes[mask_nodes] = False
        keep_nodes = keep_nodes.nonzero(as_tuple=False).view(-1)

        masked_x = self.data.x.clone()
        token_count = int(self.replace_rate * mask_nodes.numel())
        if token_count:
            permutation = torch.randperm(mask_nodes.numel(), device=self.device)
            noise_nodes = mask_nodes[permutation[:token_count]]
            token_nodes = mask_nodes[permutation[token_count:]]
            masked_x[token_nodes] = 0.0
            source = torch.randperm(num_nodes, device=self.device)[:token_count]
            masked_x[noise_nodes] = self.data.x[source]
        else:
            token_nodes = mask_nodes
            masked_x[mask_nodes] = 0.0
        masked_x[token_nodes] += self.enc_mask_token
        return masked_x, mask_nodes, keep_nodes

    def _reconstruction_loss(self, target, prediction):
        if self.loss_fn == "mse":
            return torch.nn.functional.mse_loss(prediction, target)
        return sce_loss(target, prediction, alpha=self.alpha_l)

    def _remask(self, representation):
        remask_nodes = safe_index_subset(self.data.num_nodes, self.remask_rate, self.device)
        representation = representation.clone()
        representation[remask_nodes] = self.dec_mask_token
        return representation

    @torch.no_grad()
    def _ema_update(self):
        for student, teacher in zip(self.gnn.parameters(), self.encoder_ema.parameters()):
            teacher.data.mul_(self.momentum).add_((1.0 - self.momentum) * student.detach().data)
        for student, teacher in zip(self.projector.parameters(), self.projector_ema.parameters()):
            teacher.data.mul_(self.momentum).add_((1.0 - self.momentum) * student.detach().data)

    def pretrain_one_epoch(self):
        self.gnn.train()
        self.decoder.train()
        masked_x, mask_nodes, keep_nodes = self._mask_features()
        encoder_rep = self.gnn(masked_x, self.data.edge_index)

        target_nodes = keep_nodes if keep_nodes.numel() else mask_nodes
        with torch.no_grad():
            target_rep = self.encoder_ema(self.data.x, self.data.edge_index)
            target_rep = self.projector_ema(target_rep[target_nodes])
        latent_rep = self.predictor(self.projector(encoder_rep[target_nodes]))
        loss_latent = sce_loss(latent_rep, target_rep, alpha=1)

        representation = self.encoder_to_decoder(encoder_rep)
        loss_rec = representation.new_zeros(())
        for _ in range(self.num_remasking):
            reconstruction = self.decoder(self._remask(representation), self.data.edge_index)
            loss_rec = loss_rec + self._reconstruction_loss(
                self.data.x[mask_nodes], reconstruction[mask_nodes]
            )
        loss = loss_rec + self.lam * loss_latent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._ema_update()
        return float(loss.detach().cpu())

    def pretrain(self):
        best_loss = float("inf")
        patience = 10
        stale = 0
        for epoch in range(1, self.epochs + 1):
            started = time.time()
            loss = self.pretrain_one_epoch()
            logger.info(
                "GraphMAE2 [Pretrain] Epoch %s/%s | Train Loss %.5f | Cost Time %.3fs",
                epoch,
                self.epochs,
                loss,
                time.time() - started,
            )
            if loss < best_loss:
                best_loss, stale = loss, 0
            else:
                stale += 1
                if stale >= patience:
                    break
        return save_encoder_checkpoint(
            self.gnn,
            self.dataset_name,
            "GraphMAE2",
            self.gnn_type,
            self.hid_dim,
            self.output_path,
            metadata=pretrain_metadata(self, epoch, loss),
        )
