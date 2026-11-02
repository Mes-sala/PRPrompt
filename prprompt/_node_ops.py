"""Final input-node routing and prototype readout helpers, without legacy loaders."""
import torch
from torch.nn import functional as F


def prototype_logits(z, support, y, classes, loo=False):
    z = F.normalize(z, dim=1)
    zs = z[support]
    oh = F.one_hot(y, classes).float()
    sums = oh.T @ zs
    counts = oh.sum(0)
    proto = F.normalize(sums / counts[:, None], dim=1)
    if not loo or counts.min() <= 1:
        scores = z @ proto.T / 0.2
        return scores[support] if loo else scores
    per = (sums[None] - oh[:, :, None] * zs[:, None])
    per = per / (counts[None, :, None] - oh[:, :, None])
    return (zs[:, None] * F.normalize(per, dim=2)).sum(-1) / 0.2

@torch.no_grad()
def rectified_teacher(r, support, sy, classes, seed):
    torch.manual_seed(seed)
    rn = F.normalize(r, dim=1)
    oh = F.one_hot(sy, classes).float()
    original = (oh.T @ rn[support]) / oh.sum(0)[:, None]
    centers = original.clone()
    mask = torch.ones(len(r), dtype=torch.bool, device=r.device)
    mask[support] = False
    ru = rn[mask]
    beta = 0.5
    for _ in range(3):
        assignment = (ru @ F.normalize(centers, dim=1).T / 0.05).softmax(-1)
        pseudo = (assignment.T @ ru) / assignment.sum(0).clamp_min(1e-8)[:, None]
        centers = (1 - beta) * original + beta * pseudo
    scores = rn @ F.normalize(centers, dim=1).T / 0.2
    assert torch.isfinite(scores).all() and not mask[support].any()
    return scores, {'rect_beta': beta, 'rect_iterations': 3, 'rect_unlabeled_count': int(mask.sum()), 'rect_support_overlap': int(mask[support].sum()), 'rect_center_change': (centers-original).norm().item()}


@torch.no_grad()
def temperature_route(scores, adjacency, temperature):
    if temperature <= 0:
        raise ValueError('Positive route temperature required')
    basep = scores.softmax(-1)
    probability = (scores*(.2/temperature)).softmax(-1)
    base = torch.sparse.mm(adjacency, basep-basep.mean(0, keepdim=True))
    route = torch.sparse.mm(adjacency, probability-probability.mean(0, keepdim=True))
    if not base.norm() > 0 or not route.norm() > 0:
        raise ValueError('Degenerate zero node route')
    return route, probability
