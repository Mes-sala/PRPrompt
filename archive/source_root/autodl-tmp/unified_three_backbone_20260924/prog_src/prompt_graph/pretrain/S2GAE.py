"""S2GAE-style masked-edge self-supervised pretraining for ProG."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling, to_undirected

from prompt_graph.utils import get_logger

from ._masked_utils import (
    pretrain_metadata,
    resolve_node_data,
    save_encoder_checkpoint,
    validate_rate,
)
from .base import PreTrain

logger = get_logger(__name__)


class S2GAE(PreTrain):
    """Masked edge reconstruction using the local S2GAE masking protocol.

    ``mask_type='um'`` samples edges uniformly.  ``mask_type='dm'``
    first makes the graph undirected, matching the project implementation's
    undirected node-classification branch.
    """

    def __init__(
        self,
        *args,
        data=None,
        mask_ratio=0.8,
        mask_type="dm",
        neg_ratio=1.0,
        decoder_hidden_dim=None,
        output_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data, self.input_dim, self.output_dim = resolve_node_data(
            self.dataset_name, data, self.device
        )
        self.mask_ratio = validate_rate("mask_ratio", mask_ratio, allow_zero=False)
        if mask_type not in ("um", "dm"):
            raise ValueError("mask_type must be 'um' or 'dm'")
        self.mask_type = mask_type
        if float(neg_ratio) <= 0:
            raise ValueError("neg_ratio must be positive")
        self.neg_ratio = float(neg_ratio)
        self.output_path = output_path
        self.initialize_gnn(self.input_dim, self.hid_dim)
        decoder_hidden_dim = int(decoder_hidden_dim or self.hid_dim)
        self.edge_decoder = nn.Sequential(
            nn.Linear(self.hid_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, 1),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [*self.gnn.parameters(), *self.edge_decoder.parameters()],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _mask_edges(self):
        # Author dm masks directed entries without symmetrizing the remainder.
        # Author um operates on one representative per undirected edge.
        edge_index = to_undirected(self.data.edge_index)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        if self.mask_type == "um":
            edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        num_edges = edge_index.size(1)
        if num_edges < 2:
            raise ValueError("S2GAE needs at least two maskable edges")
        mask_count = min(max(1, int(num_edges * self.mask_ratio)), num_edges - 1)
        permutation = torch.randperm(num_edges, device=self.device)
        masked = edge_index[:, permutation[-mask_count:]]
        visible = edge_index[:, permutation[:-mask_count]]
        if self.mask_type == "um":
            visible = to_undirected(visible)
        return visible, masked

    def _decode(self, node_rep, edge_index):
        edge_rep = node_rep[edge_index[0]] * node_rep[edge_index[1]]
        return self.edge_decoder(edge_rep).view(-1)

    def pretrain_one_epoch(self):
        self.gnn.train()
        self.edge_decoder.train()
        visible_edges, masked_edges = self._mask_edges()
        node_rep = self.gnn(self.data.x, visible_edges)
        negative_edges = negative_sampling(
            edge_index=self.data.edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=max(1, int(masked_edges.size(1) * self.neg_ratio)),
        ).to(self.device)
        pos_logits = self._decode(node_rep, masked_edges)
        neg_logits = self._decode(node_rep, negative_edges)
        loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        loss = loss + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def pretrain(self):
        best_loss = float("inf")
        patience = 10
        stale = 0
        for epoch in range(1, self.epochs + 1):
            started = time.time()
            loss = self.pretrain_one_epoch()
            logger.info(
                "S2GAE [Pretrain] Epoch %s/%s | Train Loss %.5f | Cost Time %.3fs",
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
            "S2GAE",
            self.gnn_type,
            self.hid_dim,
            self.output_path,
            metadata=pretrain_metadata(self, epoch, loss),
        )
