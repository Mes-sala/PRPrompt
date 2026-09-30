"""Code-faithful SUPT and GraphTOP strategies for ProG's task interfaces."""

import numpy as np
import torch
from torch.nn import functional as F

from prompt_graph.prompt.GraphTOP import (
    annealing_temperature,
    prompted_edge_weights,
    topology_losses,
)

from ..strategy import register_strategy


def classification_metrics(labels, logits):
    from prompt_graph.evaluation.node_scores import classification_scores
    return classification_scores(torch.cat(logits), torch.cat(labels))


@register_strategy("SUPT-soft")
class SUPTSoftStrategy:
    def train_epoch(self, ctx, loader):
        ctx.gnn.eval()
        ctx.prompt.train()
        ctx.answering.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(ctx.device)
            ctx.optimizer.zero_grad()
            x = ctx.prompt.add(batch.x, batch.edge_index, batch.batch)
            logits = ctx.answering(ctx.gnn(x, batch.edge_index, batch.batch))
            loss = F.cross_entropy(logits, batch.y.view(-1))
            if ctx.prompt.orth_loss and ctx.prompt.num_clusters > 1:
                loss = loss + ctx.prompt.orthogonal_loss()
            loss.backward()
            ctx.optimizer.step()
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
        if count == 0:
            raise ValueError("SUPT needs a nonempty support loader")
        return total / count

    @torch.no_grad()
    def evaluate(self, ctx, loader):
        ctx.gnn.eval()
        ctx.prompt.eval()
        ctx.answering.eval()
        labels, logits = [], []
        for batch in loader:
            batch = batch.to(ctx.device)
            x = ctx.prompt.add(batch.x, batch.edge_index, batch.batch)
            logits.append(ctx.answering(ctx.gnn(x, batch.edge_index, batch.batch)))
            labels.append(batch.y)
        return classification_metrics(labels, logits)


@register_strategy("SUPT-hard")
class SUPTHardStrategy(SUPTSoftStrategy):
    pass


@register_strategy("GraphTOP")
class GraphTOPStrategy:
    def train_epoch(self, ctx, loader):
        ctx.gnn.eval()
        ctx.prompt.train()
        ctx.answering.train()
        tau = annealing_temperature(ctx.extra["epoch"], ctx.extra["epochs"])
        total = []
        for batch in loader:
            batch = batch.to(ctx.device)
            ctx.optimizer.zero_grad()
            if ctx.extra.get("disable_prompt", False):
                weights, prob = batch.edge_weight, None
            else:
                weights, prob = prompted_edge_weights(
                    ctx.prompt, ctx.extra["embeddings"], batch, tau, True
                )
            emb = ctx.gnn(batch.x, batch.edge_index, edge_weight=weights)
            logits = ctx.answering(emb[batch.target_node_index])
            ce = F.cross_entropy(logits, batch.y.view(-1))
            entropy, sparsity = topology_losses(prob, batch) if prob is not None else (0.0, 0.0)
            loss = (
                ce
                + ctx.extra.get("lambda_e", 2.0) * entropy
                + ctx.extra.get("lambda_s", 10.0) * sparsity
            )
            loss.backward()
            ctx.optimizer.step()
            total.append(loss.item())
        if not total:
            raise ValueError("GraphTOP needs a nonempty support loader")
        return float(np.mean(total))  # Upstream averages minibatch losses equally.

    @torch.no_grad()
    def evaluate(self, ctx, loader):
        ctx.gnn.eval()
        ctx.prompt.eval()
        ctx.answering.eval()
        labels, logits = [], []
        for batch in loader:
            batch = batch.to(ctx.device)
            if ctx.extra.get("disable_prompt", False):
                weights = batch.edge_weight
            else:
                weights, _ = prompted_edge_weights(
                    ctx.prompt, ctx.extra["embeddings"], batch, 0.03, False
                )
            emb = ctx.gnn(batch.x, batch.edge_index, edge_weight=weights)
            logits.append(ctx.answering(emb[batch.target_node_index]))
            labels.append(batch.y)
        return classification_metrics(labels, logits)
