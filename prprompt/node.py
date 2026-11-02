"""Final input-layer node PRPrompt: fixed route, trainable C x D tokens.

Both APIs accept only support labels. Full-graph features/routing and output
centering are transductive. Subgraph construction is supplied by the caller.
"""
import copy
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from .backbone import normalized_adjacency
from .utils import support_vectors, state_hash
from ._node_ops import rectified_teacher, prototype_logits, temperature_route

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



def fit_fullgraph(backbone, x, edge_index, support, support_labels, classes, *, seed=42, steps=100):
    """Final F100Soft input recipe, independent of archived experiments.

    100 Adam steps/.01, tanh CxD tokens, input amplitude10, .001 token L2,
    support prototype CE (.2), LOO for >1-shot and self-prototype for 1-shot.
    Returns all-node scores, embeddings, prompt state and audit information.
    """
    if x.ndim != 2 or len(x) < 2 or not torch.isfinite(x).all() or steps < 1:
        raise ValueError('Finite [N,D] features with N>=2 and positive steps required')
    support, sy = support_vectors(support, support_labels, classes, len(x), x.device)
    model = copy.deepcopy(backbone).to(x.device).eval().requires_grad_(False)
    before = state_hash(model); edge_index = edge_index.to(x.device)
    adj = normalized_adjacency(edge_index, len(x), x.dtype)
    with torch.no_grad():
        r = torch.sparse.mm(adj, torch.sparse.mm(adj, x)).detach()
        raw, _ = rectified_teacher(r-r.mean(0, keepdim=True), support, sy, classes, seed)
        gate, probability = temperature_route(raw, adj, .1)
        scale = x.std(0).clamp_min(1e-6).detach()
    tokens = torch.nn.Parameter(x.new_zeros(classes, x.shape[1]))
    opt = torch.optim.Adam([tokens], lr=.01)
    def forward():
        prompted = x + 10*(gate @ tokens.tanh())*scale
        return model.conv_layers[1](F.leaky_relu(model.conv_layers[0](prompted, edge_index)), edge_index)
    gradients, trace = [], []
    for step in range(steps):
        opt.zero_grad(); z = forward()
        ce = F.cross_entropy(prototype_logits(z, support, sy, classes, loo=True), sy)
        loss = ce + .001*tokens.square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(tokens.grad).all():
            raise RuntimeError('Nonfinite prompt fit')
        gradients.append(tokens.grad.detach().cpu().clone()); opt.step()
        trace.append(dict(step=step, loss=float(loss.detach()), ce=float(ce.detach()), gradient_norm=float(tokens.grad.norm())))
    with torch.no_grad():
        z = forward().detach()
        scores = prototype_logits(z-z.mean(0, keepdim=True), support, sy, classes)
    if state_hash(model) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen backbone changed')
    state = dict(tokens=tokens.detach().cpu(), gate=gate.cpu(), scale=scale.cpu(),
                 probability=probability.cpu(), optimizer_gradients=torch.stack(gradients), trace=trace)
    return scores, z, state, dict(backbone_frozen=True, query_labels_received=False, injection_layer='input',
             prompt_parameters=tokens.numel(), steps=steps, one_shot_loss_adapted=bool(torch.bincount(sy).min()==1))


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
    torch.manual_seed(cfg.seed)
    edge_index = edge_index.to(x.device)
    adj = normalized_adjacency(edge_index, len(x), x.dtype)
    with torch.no_grad():
        r = torch.sparse.mm(adj, torch.sparse.mm(adj, x))
        r = r - r.mean(0, keepdim=True)
        raw = rectified_teacher(r, support, support_labels, classes, cfg.seed)[0]
        gate, _ = temperature_route(raw, adj, cfg.route_temperature)
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
        scores = prototype_logits(z, local_support, support_labels, classes, loo=loo)
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
        scores = prototype_logits(centered, support, support_labels, classes)
    unchanged = all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    if not unchanged or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen backbone changed or accumulated gradients')
    return scores, z, {'tokens': tokens.detach(), 'gate': gate, 'trace': trace}, {
        'backbone_frozen': unchanged, 'injection_layer': 'input', 'readout': 'subgraph_mean_prototype',
        'prompt_parameters': tokens.numel(), 'support_class_counts': counts.cpu().tolist(),
        'leave_one_out_training': loo, 'query_labels_received': False,
        'subgraph_label_provenance': 'caller_responsibility', 'routing': 'fullgraph_fixed_soft',
    }
