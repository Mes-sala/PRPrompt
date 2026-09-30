"""Frozen node prototypes, with full graph or self-loop-only matched context."""
import torch
import torch.nn.functional as F

from prompt_graph.evaluation.node_scores import classification_scores

from ..strategy import register_strategy
from .self_pro import _compute_centers


@register_strategy('FrozenNodePrototype')
@register_strategy('FrozenIdentityPrototype')
class FrozenNodePrototype:
    def train_epoch(self, ctx, loader_or_data):
        return 0.0

    @torch.no_grad()
    def predict_logits(self, ctx, data, query_idx):
        ctx.gnn.eval()
        edges = data.edge_index
        if ctx.extra.get('identity_graph'):
            edges = torch.arange(data.num_nodes, device=ctx.device).repeat(2, 1)
        z = F.normalize(ctx.gnn(data.x, edges), p=2, dim=1)
        centers = _compute_centers(z[ctx.extra['support_idx']], ctx.extra['support_labels'], ctx.output_dim)
        return z[query_idx] @ F.normalize(centers, p=2, dim=1).t()

    def evaluate(self, ctx, loader_or_data):
        data, query = loader_or_data
        return classification_scores(self.predict_logits(ctx, data, query), data.y[query])
