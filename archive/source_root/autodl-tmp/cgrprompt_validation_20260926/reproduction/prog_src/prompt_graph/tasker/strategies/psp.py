"""PSPStrategy: prompt_type == 'PSP' train/eval logic.

PSP (ICLR 2024) learns structural prompt weights between nodes and label
prototypes.  This ProG port replaces the dual pre-trained encoders with the
standard GNN backbone and uses a simplified prototype-similarity classifier.
"""

from __future__ import annotations

import torch.nn.functional as F

import torch
from prompt_graph.evaluation.node_scores import classification_scores

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("PSP")
class PSPStrategy(PromptStrategy):
    """Prototype-structured prompting."""

    def setup(self, ctx: TaskContext) -> None:
        """No-op: prompt is built by BaseTask.initialize_prompt."""

    def configure_optimizer(self, ctx: TaskContext) -> None:
        """No-op: optimizer is built by BaseTask.initialize_optimizer."""

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        data, train_idx = loader_or_data
        ctx.prompt.train()
        ctx.optimizer.zero_grad()
        logits = ctx.prompt(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        ctx.optimizer.step()
        return loss.item()

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        data, idx_test = loader_or_data
        ctx.prompt.eval()
        with torch.no_grad():
            logits = ctx.prompt(data.x, data.edge_index)[idx_test]
        return classification_scores(logits, data.y[idx_test])
