"""Dual-Harmonic Boundary Prompt modules.

DHBP builds an input-space prompt field from a small set of labelled boundary
nodes, then injects it through a zero-initialised residual gate.  The first
implementation targets NodeTask, matching the research report's P0 scope.
"""

from __future__ import annotations

import torch
from torch import nn


class DHBPPrompt(nn.Module):
    """Residual, routed L/Q harmonic boundary prompt for node classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_iters: int = 8,
        mu: float = 0.1,
        max_intervention: float = 0.5,
    ):
        super().__init__()
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1")
        if mu <= 0:
            raise ValueError("mu must be positive")
        if max_intervention <= 0:
            raise ValueError("max_intervention must be positive")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_iters = num_iters
        self.mu = mu
        self.max_intervention = max_intervention

        self.boundary_plus = nn.Parameter(torch.zeros(num_classes, in_channels))
        self.boundary_minus = nn.Parameter(torch.zeros(num_classes, in_channels))
        self.router = nn.Linear(in_channels + 1, 2)
        self.inject_logit = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

        self.last_diagnostics: dict[str, float] = {}

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.boundary_plus)
        nn.init.xavier_uniform_(self.boundary_minus)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        nn.init.zeros_(self.inject_logit)

    @property
    def injection_scale(self) -> torch.Tensor:
        return torch.tanh(self.inject_logit)

    def add(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        boundary_idx: torch.Tensor,
        boundary_y: torch.Tensor,
    ) -> torch.Tensor:
        field_plus, field_minus, degree = self._solve_fields(
            x, edge_index, boundary_idx, boundary_y
        )
        route = self._route(x, degree)
        field = route[:, :1] * field_plus + route[:, 1:] * field_minus

        scale = self.injection_scale
        delta = scale * field
        delta = self._cap_intervention(x, delta)
        self.last_diagnostics = self._diagnostics(
            x, delta, field_plus, field_minus, edge_index, boundary_idx, boundary_y
        )
        return x + delta

    def _solve_fields(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        boundary_idx: torch.Tensor,
        boundary_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        device = x.device
        boundary_idx = boundary_idx.to(device)
        boundary_y = boundary_y.to(device)

        degree = self._degree(edge_index, num_nodes, device)
        boundary_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        boundary_mask[boundary_idx] = True

        field_plus = x.new_zeros(num_nodes, self.in_channels)
        field_minus = x.new_zeros(num_nodes, self.in_channels)
        if boundary_idx.numel() > 0:
            field_plus[boundary_idx] = self.boundary_plus[boundary_y]
            field_minus[boundary_idx] = self.boundary_minus[boundary_y]

        for _ in range(self.num_iters):
            agg_plus = self._aggregate(field_plus, edge_index, degree)
            agg_minus = self._aggregate(field_minus, edge_index, degree)
            next_plus = agg_plus / (1.0 + self.mu)
            next_minus = -agg_minus / (1.0 + self.mu)
            field_plus = torch.where(boundary_mask.unsqueeze(1), field_plus, next_plus)
            field_minus = torch.where(boundary_mask.unsqueeze(1), field_minus, next_minus)

        return field_plus, field_minus, degree

    def _route(self, x: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        log_degree = torch.log1p(degree).unsqueeze(1)
        logits = self.router(torch.cat([x, log_degree], dim=1))
        return torch.softmax(logits, dim=1)

    @staticmethod
    def _degree(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        dst = edge_index[1].to(device)
        degree = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        return degree.clamp_min_(1.0)

    @staticmethod
    def _aggregate(
        field: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index.to(field.device)
        out = field.new_zeros(field.shape)
        msg = field[src] / degree[dst].unsqueeze(1)
        out.index_add_(0, dst, msg)
        return out

    def _cap_intervention(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x_norm = torch.linalg.vector_norm(x.detach()).clamp_min(1e-12)
        delta_norm = torch.linalg.vector_norm(delta).clamp_min(1e-12)
        ratio = (self.max_intervention * x_norm / delta_norm).clamp(max=1.0)
        return delta * ratio

    def _diagnostics(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        field_plus: torch.Tensor,
        field_minus: torch.Tensor,
        edge_index: torch.Tensor,
        boundary_idx: torch.Tensor,
        boundary_y: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            x_norm = torch.linalg.vector_norm(x).clamp_min(1e-12)
            delta_norm = torch.linalg.vector_norm(delta)
            r_l = self._relative_residual(field_plus, edge_index, boundary_idx, boundary_y, "L")
            r_q = self._relative_residual(field_minus, edge_index, boundary_idx, boundary_y, "Q")
            return {
                "dhbp_r_l": float(r_l.detach().cpu()),
                "dhbp_r_q": float(r_q.detach().cpu()),
                "dhbp_intervention_ratio": float((delta_norm / x_norm).detach().cpu()),
                "dhbp_injection_scale": float(self.injection_scale.detach().cpu()),
            }

    def _relative_residual(
        self,
        field: torch.Tensor,
        edge_index: torch.Tensor,
        boundary_idx: torch.Tensor,
        boundary_y: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        num_nodes = field.size(0)
        device = field.device
        boundary_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        boundary_mask[boundary_idx.to(device)] = True
        interior_mask = ~boundary_mask
        if not torch.any(interior_mask):
            return field.new_tensor(0.0)

        degree = self._degree(edge_index, num_nodes, device)
        agg = self._aggregate(field, edge_index, degree)
        residual = (1.0 + self.mu) * field - agg
        if mode == "Q":
            residual = (1.0 + self.mu) * field + agg

        numerator = torch.linalg.vector_norm(residual[interior_mask])
        boundary = self.boundary_plus if mode == "L" else self.boundary_minus
        target = boundary[boundary_y.to(device)] if boundary_y.numel() else field.new_zeros(1)
        denominator = torch.linalg.vector_norm(target).clamp_min(1e-12)
        return numerator / denominator
