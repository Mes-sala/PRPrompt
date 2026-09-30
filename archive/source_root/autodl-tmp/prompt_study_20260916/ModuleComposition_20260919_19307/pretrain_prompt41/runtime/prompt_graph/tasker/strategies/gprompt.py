"""GraphPrompt: cosine loss and final-state support prototypes (protocol v3)."""
import torch

from prompt_graph.evaluation import GpromptEva
from prompt_graph.utils import Gprompt_tuning_loss, center_embedding
from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy('Gprompt')
class GpromptStrategy(PromptStrategy):
    def __init__(self):
        self.mean_centers = None
        self.support_loader = None

    def train_epoch(self, ctx: TaskContext, train_loader) -> float:
        ctx.gnn.eval()
        ctx.prompt.train()
        self.support_loader = train_loader
        self.mean_centers = None
        total_loss, examples = 0.0, 0
        criterion = Gprompt_tuning_loss()
        for batch in train_loader:
            if ctx.pg_opi is not None:
                ctx.pg_opi.zero_grad()
            batch = batch.to(ctx.device)
            out = ctx.gnn(batch.x, batch.edge_index, batch.batch,
                          prompt=ctx.prompt, prompt_type='Gprompt')
            center, _ = center_embedding(out, batch.y, ctx.output_dim)
            loss = criterion(out, center, batch.y)
            if ctx.pg_opi is not None:
                loss.backward()
                ctx.pg_opi.step()
            total_loss += loss.item() * batch.y.numel()
            examples += batch.y.numel()
        if not examples:
            raise ValueError('Empty support loader')
        return total_loss / examples

    @torch.no_grad()
    def refresh_centers(self, ctx):
        if self.support_loader is None:
            raise ValueError('Support loader must be set before evaluation')
        ctx.gnn.eval()
        ctx.prompt.eval()
        sums = counts = None
        for batch in self.support_loader:
            batch = batch.to(ctx.device)
            out = ctx.gnn(batch.x, batch.edge_index, batch.batch,
                          prompt=ctx.prompt, prompt_type='Gprompt')
            if sums is None:
                sums = out.new_zeros(ctx.output_dim, out.size(1))
                counts = out.new_zeros(ctx.output_dim)
            sums.index_add_(0, batch.y, out)
            counts.index_add_(0, batch.y, out.new_ones(batch.y.numel()))
        if counts is None or (counts == 0).any():
            raise ValueError('Every class needs support for prototype evaluation')
        self.mean_centers = sums / counts[:, None]
        return self.mean_centers

    def evaluate(self, ctx: TaskContext, test_loader) -> tuple:
        self.refresh_centers(ctx)
        return GpromptEva(test_loader, ctx.gnn, ctx.prompt, self.mean_centers,
                          ctx.output_dim, ctx.device)


@register_strategy('FrozenGraphPrototype')
class FrozenGraphPrototypeStrategy(GpromptStrategy):
    """Same induced graph and prototype readout, no learned prompt."""
