"""DHBPStrategy: prompt_type == 'DHBP' train/eval logic."""

from __future__ import annotations

import torch
import torchmetrics

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("DHBP")
class DHBPStrategy(PromptStrategy):
    """NodeTask-only DHBP adapter with frozen encoder and trainable head."""

    def train_epoch(self, ctx: TaskContext, data_and_idx) -> float:
        data, train_idx = data_and_idx
        self._freeze_encoder(ctx)
        ctx.prompt.train()
        ctx.answering.train()
        ctx.optimizer.zero_grad()

        data = data.to(ctx.device)
        x_prompted = ctx.prompt.add(data.x, data.edge_index, train_idx, data.y[train_idx])
        out = ctx.gnn(x_prompted, data.edge_index, batch=None)
        out = ctx.answering(out)
        loss = ctx.criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        ctx.optimizer.step()
        ctx.extra["dhbp_last_diagnostics"] = dict(ctx.prompt.last_diagnostics)
        return loss.item()

    def evaluate(self, ctx: TaskContext, data_and_idx) -> tuple:
        data, idx_test, train_idx = data_and_idx
        self._freeze_encoder(ctx)
        ctx.prompt.eval()
        ctx.answering.eval()

        accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=ctx.output_dim
        ).to(ctx.device)
        macro_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=ctx.output_dim, average="macro"
        ).to(ctx.device)
        auroc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=ctx.output_dim
        ).to(ctx.device)
        auprc = torchmetrics.classification.AveragePrecision(
            task="multiclass", num_classes=ctx.output_dim
        ).to(ctx.device)

        data = data.to(ctx.device)
        with torch.no_grad():
            x_prompted = ctx.prompt.add(data.x, data.edge_index, train_idx, data.y[train_idx])
            out = ctx.gnn(x_prompted, data.edge_index, batch=None)
            out = ctx.answering(out)
            pred = out.argmax(dim=1)

        acc = accuracy(pred[idx_test], data.y[idx_test])
        f1 = macro_f1(pred[idx_test], data.y[idx_test])
        roc = auroc(out[idx_test], data.y[idx_test])
        prc = auprc(out[idx_test], data.y[idx_test])
        ctx.extra["dhbp_last_diagnostics"] = dict(ctx.prompt.last_diagnostics)
        return acc.item(), f1.item(), roc.item(), prc.item()

    @staticmethod
    def _freeze_encoder(ctx: TaskContext) -> None:
        ctx.gnn.eval()
        for param in ctx.gnn.parameters():
            param.requires_grad = False
