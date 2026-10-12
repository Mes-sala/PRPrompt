"""Input-layer F100Soft and subgraph-input PRPrompt, independent of task samplers.

Run from a repository checkout: the full-graph entry intentionally retains the
archived F100Soft implementation and its numerical checks.
"""
import copy
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from prprompt.model import normalized_adjacency


@dataclass(frozen=True)
class SubgraphConfig:
    steps: int = 100
    lr: float = 0.01
    weight_decay: float = 0.0
    regularization: float = 0.001
    amplitude: float = 10.0
    route_temperature: float = 0.1
    seed: int = 42
    inference_batch_size: int = 64


def fit_fullgraph(backbone, x, edge_index, support, support_labels, classes,
                  *, seed=42, steps=100):
    """Exact archived input Soft entry; return scores, embeddings, state, audit."""
    from experiments.prog_split_20261002.numeric_diagnostics import install
    from experiments.f100_input_20261005.model import fit_input
    install()
    return fit_input('Soft', backbone, x, edge_index, support, support_labels,
                     classes, seed, steps)


def fit_subgraph(backbone, x, edge_index, support, support_labels, classes,
                 subgraph_for_center, *, config=None):
    """Train input prompts with mean-pooled subgraphs and prototype readout.

    subgraph_for_center(center_id) returns a mapping with `nodes` (unique global
    node IDs, including the center) and `edges` (locally relabeled edge_index).
    This API neither constructs subgraphs nor samples tasks. It receives only
    support labels. Callers must ensure graph construction does not use held-out
    labels. Scores have one row per global center; slice an explicit disjoint
    query index outside this function. Routing remains full-graph transductive.
    """
    from experiments.v1_f100.historical import load
    cfg = config or SubgraphConfig()
    if cfg.steps < 1 or cfg.inference_batch_size < 1 or cfg.route_temperature <= 0:
        raise ValueError('steps, inference batch size and temperature must be positive')
    if x.ndim != 2 or len(x) < 2 or not torch.isfinite(x).all():
        raise ValueError('x must be a finite feature matrix with at least two nodes')
    support = support.to(device=x.device, dtype=torch.long)
    support_labels = support_labels.to(device=x.device, dtype=torch.long)
    if support.ndim != 1 or support_labels.shape != support.shape or not len(support):
        raise ValueError('support and support_labels must be nonempty aligned vectors')
    if (support < 0).any() or (support >= len(x)).any() or len(support.unique()) != len(support):
        raise ValueError('support must contain unique valid node IDs')
    if (support_labels < 0).any() or (support_labels >= classes).any():
        raise ValueError('support labels must be in [0, classes)')
    counts = torch.bincount(support_labels, minlength=classes)
    if (counts == 0).any():
        raise ValueError('Each class must have at least one support node')
    model = copy.deepcopy(backbone).to(x.device).eval().requires_grad_(False)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ns = load()
    torch.manual_seed(cfg.seed)
    edge_index = edge_index.to(x.device)
    adj = normalized_adjacency(edge_index, len(x), x.dtype)
    with torch.no_grad():
        r = torch.sparse.mm(adj, torch.sparse.mm(adj, x))
        r = r - r.mean(0, keepdim=True)
        raw = ns['rectified_teacher'](r, support, support_labels, classes,
                                       'rect_teacher_ref50', cfg.seed)[0]
        gate, _ = ns['temperature_gate'](raw, adj, cfg.route_temperature, 'raw')
        gate = gate.detach()
        scale = x.std(0).clamp_min(1e-6).detach()
    xcpu, gate_cpu = x.detach().cpu(), gate.cpu()

    def make_batch(ids):
        graphs = []
        for center in ids:
            center = int(center)
            item = subgraph_for_center(center)
            nodes = item['nodes'].detach().cpu().long()
            edges = item['edges'].detach().cpu().long()
            if nodes.ndim != 1 or not len(nodes) or len(nodes.unique()) != len(nodes):
                raise ValueError('Subgraph nodes must be a nonempty vector of unique IDs')
            if (nodes < 0).any() or (nodes >= len(x)).any() or not (nodes == center).any():
                raise ValueError('Invalid subgraph node IDs or missing center')
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError('Subgraph edges must have shape [2, E]')
            if edges.numel() and ((edges < 0).any() or (edges >= len(nodes)).any()):
                raise ValueError('Subgraph edges must use local node IDs')
            graphs.append(Data(x=xcpu[nodes], edge_index=edges, route=gate_cpu[nodes]))
        return Batch.from_data_list(graphs).to(x.device)

    tokens = torch.nn.Parameter(x.new_zeros(classes, x.shape[1]))
    optimizer = torch.optim.Adam([tokens], lr=cfg.lr, weight_decay=cfg.weight_decay)
    support_batch = make_batch(support.detach().cpu().tolist())
    local_support = torch.arange(len(support), device=x.device)
    loo = bool(counts.min() > 1)

    def forward(batch):
        prompted = batch.x + cfg.amplitude * (batch.route @ tokens.tanh()) * scale
        node_z = model(prompted, batch.edge_index)
        return global_mean_pool(node_z, batch.batch)

    trace = []
    for step in range(cfg.steps):
        optimizer.zero_grad()
        z = forward(support_batch)
        scores = ns['logits'](z, local_support, support_labels, classes, loo=loo)
        ce = F.cross_entropy(scores, support_labels)
        reg = cfg.regularization * tokens.square().mean()
        loss = ce + reg
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite prompt loss')
        loss.backward()
        if not torch.isfinite(tokens.grad).all():
            raise RuntimeError('Non-finite prompt gradient')
        optimizer.step()
        trace.append(dict(step=step, loss=float(loss.detach()), ce=float(ce.detach()),
                          regularizer=float(reg.detach())))
    with torch.no_grad():
        embeddings = []
        for start in range(0, len(x), cfg.inference_batch_size):
            embeddings.append(forward(make_batch(range(start, min(start + cfg.inference_batch_size, len(x))))).cpu())
        z = torch.cat(embeddings).to(x.device)
        centered = z - z.mean(0, keepdim=True)
        scores = ns['logits'](centered, support, support_labels, classes)
    unchanged = all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    if not unchanged or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen backbone changed or accumulated gradients')
    return scores, z, {'tokens': tokens.detach(), 'gate': gate, 'trace': trace}, {
        'backbone_frozen': unchanged, 'injection_layer': 'input', 'readout': 'subgraph_mean_prototype',
        'prompt_parameters': tokens.numel(), 'support_class_counts': counts.cpu().tolist(),
        'leave_one_out_training': loo, 'query_labels_received': False,
        'subgraph_label_provenance': 'caller_responsibility', 'routing': 'fullgraph_fixed_soft',
    }
