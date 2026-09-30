"""Bandana's non-discrete bandwidth masking port for ProG.

The local Bandana project computes a temperature-controlled softmax over
outgoing edges at every encoder layer and trains a link decoder with the
resulting continuous bandwidths.  The port keeps that objective and uses
ProG's GCN layers so the exported encoder is immediately compatible with the
rest of ProG.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling, softmax

from prompt_graph.utils import act, get_logger

from ._masked_utils import pretrain_metadata, resolve_node_data, save_encoder_checkpoint
from .base import PreTrain

logger = get_logger(__name__)


class BandanaEdgeDecoder(nn.Module):
    def __init__(self, hid_dim, decoder_hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hid_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, 1),
        )

    def forward(self, node_rep, edge_index):
        edge_rep = node_rep[edge_index[0]] * node_rep[edge_index[1]]
        return self.layers(edge_rep).view(-1)


class Bandana(PreTrain):
    """Full-batch Bandana pretraining for GCN backbones."""

    def __init__(
        self,
        *args,
        data=None,
        temp=1.0,
        neg_ratio=0.5,
        decoder_hidden_dim=None,
        output_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.gnn_type != "GCN":
            raise ValueError(
                "Bandana currently requires gnn_type='GCN' for edge weighting fidelity"
            )
        self.data, self.input_dim, self.output_dim = resolve_node_data(
            self.dataset_name, data, self.device
        )
        self.temp = self._normalize_temperature(temp)
        if float(neg_ratio) <= 0:
            raise ValueError("neg_ratio must be positive")
        self.neg_ratio = float(neg_ratio)
        self.output_path = output_path
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.edge_decoder = BandanaEdgeDecoder(
            self.hid_dim, int(decoder_hidden_dim or self.hid_dim)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [*self.gnn.parameters(), *self.edge_decoder.parameters()],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _normalize_temperature(self, temp):
        if isinstance(temp, (list, tuple)):
            values = [float(value) for value in temp]
        else:
            values = [float(temp)]
        if not values or any(value <= 0 for value in values):
            raise ValueError("temp must contain positive values")
        if len(values) == 1:
            values *= self.num_layer
        if len(values) != self.num_layer:
            raise ValueError("temp must be a scalar or have one value per GNN layer")
        return values

    def _bandwidth(self, edge_index, temperature):
        logits = torch.randn(edge_index.size(1), device=self.device)
        return softmax(logits / temperature, edge_index[0], num_nodes=self.data.num_nodes)

    def _encode_with_bandwidth(self, edge_index):
        representation = self.data.x
        layer_outputs = []
        for layer_id, conv in enumerate(self.gnn.conv_layers):
            bandwidth = self._bandwidth(edge_index, self.temp[layer_id])
            representation = conv(representation, edge_index, bandwidth)
            if layer_id != len(self.gnn.conv_layers) - 1:
                representation = act(representation)
                representation = F.dropout(
                    representation, self.gnn.drop_ratio, training=self.training
                )
            layer_outputs.append((representation, bandwidth))
        return layer_outputs

    def _layer_loss(self, node_rep, bandwidth, pos_edges, neg_edges):
        pos_logits = self.edge_decoder(node_rep, pos_edges)
        neg_logits = self.edge_decoder(node_rep, neg_edges)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), weight=bandwidth
        )
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        return pos_loss + neg_loss

    def pretrain_one_epoch(self):
        self.gnn.train()
        self.edge_decoder.train()
        edge_index = self.data.edge_index
        negative_edges = negative_sampling(
            edge_index=edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=max(1, int(edge_index.size(1) * self.neg_ratio)),
        ).to(self.device)
        losses = []
        for node_rep, bandwidth in self._encode_with_bandwidth(edge_index):
            losses.append(self._layer_loss(node_rep, bandwidth, edge_index, negative_edges))
        loss = torch.stack(losses).mean()
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
                "Bandana [Pretrain] Epoch %s/%s | Train Loss %.5f | Cost Time %.3fs",
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
            "Bandana",
            self.gnn_type,
            self.hid_dim,
            self.output_path,
            metadata=pretrain_metadata(self, epoch, loss),
        )
