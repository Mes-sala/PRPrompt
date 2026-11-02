"""Locked global-statistics PRPrompt: reference -> fixed route -> input tokens.

Only support labels enter routing/fitting. Prediction uses support prototypes
recomputed after training, with the selector's inference gain applied to both
support and query inputs. Graph IDs refer to the explicit reference collection.
"""
import copy
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from .selection import choose, describe, bandwidth, centered
from .views import build_view_routes
from .utils import seed_everything, support_vectors, state_hash


def clean_graphs(graphs):
    out = []
    for graph in graphs:
        x, edges = graph.x.detach().cpu(), graph.edge_index.detach().cpu().long()
        if x.ndim != 2 or not len(x) or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Each graph needs a nonempty finite floating point x')
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError('edge_index must be [2,E]')
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= len(x)):
            raise ValueError('edge_index outside graph')
        out.append(Data(x=x, edge_index=edges))
    if not out or len({g.x.shape[1] for g in out}) != 1:
        raise ValueError('Nonempty collection with a common feature dimension required')
    return out


def batch_descriptors(x, edges, batch):
    x = x.double()
    count = torch.bincount(batch).to(x.dtype)
    degree = torch.bincount(edges[0], minlength=len(x)).to(x.dtype)
    degree_mean = global_mean_pool(degree[:, None], batch).squeeze(1)
    degree_std = global_mean_pool((degree-degree_mean[batch]).square()[:, None], batch).squeeze(1).sqrt()
    mean = global_mean_pool(x, batch)
    std = global_mean_pool((x-mean[batch]).square(), batch).sqrt()
    return torch.cat([torch.stack([count, degree_mean, degree_std], dim=1).log1p(), mean, std], dim=1)


@torch.no_grad()
def descriptors(graphs, batch_size=128):
    rows = []
    for start in range(0, len(graphs), batch_size):
        b = Batch.from_data_list(graphs[start:start+batch_size])
        rows.append(batch_descriptors(b.x, b.edge_index, b.batch))
    return torch.cat(rows)


@dataclass
class GraphReference:
    mean: torch.Tensor
    std: torch.Tensor
    h: torch.Tensor
    nodes: torch.Tensor
    selection: dict

    def state_dict(self):
        return dict(mean=self.mean, std=self.std, h=self.h, nodes=self.nodes, selection=self.selection)

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


@torch.no_grad()
def prepare_graph_reference(graphs, *, stats_state=None):
    """Calibrate the declared unlabelled pool and run the frozen selection rule.

    stats_state optionally supplies exact historical descriptor_mean/std. No
    graph label field is read. Including unlabelled query graphs is transductive.
    """
    graphs = clean_graphs(graphs)
    if len(graphs) < 3:
        raise ValueError('Global reconstruction selection requires at least three graphs')
    raw = descriptors(graphs)
    if stats_state is None:
        mean, std = raw.mean(0), raw.std(0, unbiased=False).clamp_min(1e-6)
    else:
        ref = stats_state.get('reference', stats_state)
        mean, std = [torch.as_tensor(ref[k], dtype=torch.float64).cpu() for k in ['descriptor_mean','descriptor_std']]
        if mean.shape != raw.shape[1:] or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid saved statistics reference')
    h = (raw-mean)/std
    replicas = []
    for seed in [2026110101, 2026110102]:
        gen = torch.Generator().manual_seed(seed)
        rows = []
        for g in graphs:
            x = g.x.double(); n = len(x)
            degree = torch.bincount(g.edge_index[0], minlength=n).double()
            take = torch.randint(n, (n,), generator=gen); xb, db = x[take], degree[take]
            rows.append(torch.cat([torch.tensor([n, db.mean(), db.std(unbiased=False)], dtype=torch.float64).log1p(), xb.mean(0), xb.std(0, unbiased=False)]))
        replicas.append((torch.stack(rows)-mean)/std)
    restored = h*std+mean
    active = restored.std(0, unbiased=False) > 64*torch.finfo(torch.float32).eps*restored.square().mean(0).sqrt().clamp_min(1)
    allx = torch.cat([g.x for g in graphs]).double()
    within = torch.stack([g.x.double().var(0, unbiased=False) for g in graphs]).mean(0)
    ratio = max(1., float(allx.square().mean().sqrt()/within.mean().sqrt().clamp_min(1e-12)))
    nodes = torch.tensor([len(g.x) for g in graphs], dtype=torch.float64)
    selection = choose(h, replicas, nodes, active, ratio, describe([g.x for g in graphs]))
    return GraphReference(mean, std, h, nodes, selection)


@torch.no_grad()
def make_graph_routes(graphs, support, labels, classes, reference, config):
    graphs = clean_graphs(graphs)
    si, sy = support_vectors(support, labels, classes, len(graphs), device='cpu')
    h, nodes = reference.h, reference.nodes
    if len(h) != len(graphs) or h.shape[1] != 3+2*graphs[0].x.shape[1]:
        raise ValueError('Reference collection mismatch')
    sb = Batch.from_data_list([graphs[int(i)] for i in si])
    scale = sb.x.std(0, unbiased=False).clamp_min(1e-6)
    family = config['family']
    if family == 'ViewMeanStats':
        _, table, _ = build_view_routes(h, si, sy, sb.batch, graphs[0].x.shape[1], classes)
        p = table.softmax(-1)
    elif family == 'StatsRoute':
        hs = (batch_descriptors(sb.x, sb.edge_index, sb.batch)-reference.mean)/reference.std
        q = F.normalize(torch.stack([hs[sy == c].mean(0) for c in range(classes)]), dim=-1)
        p = (F.normalize(h, dim=-1) @ q.T / .1).softmax(-1)
        ps = (F.normalize(hs, dim=-1) @ q.T / .1).softmax(-1)
        center = ps.mean(0)
        rms = (ps-center)[sb.batch].square().mean().sqrt().clamp_min(1e-6)
        table = (p-center)/rms
    elif family == 'RawKernelStats':
        kernel = (-torch.cdist(h, h).square()/bandwidth(h)).exp()
        score = torch.stack([kernel[:, si[sy == c]].mean(1) for c in range(classes)], 1)
        p = (score/.1).softmax(-1)
        table = centered(p, si, nodes)
    else:
        raise ValueError(f'Unknown final routing family: {family}')
    if config['hard']:
        p = F.one_hot(p.argmax(1), classes).double()
        table = centered(p, si, nodes)
    return table, scale, p


class GraphInputPrompt(nn.Module):
    def __init__(self, table, scale):
        super().__init__()
        self.tokens = nn.Parameter(torch.zeros(table.shape[1], len(scale), device=scale.device))
        self.register_buffer('table', table)
        self.register_buffer('scale', scale)

    def inject(self, x, ids, gain=1):
        return x + ((self.table[ids] @ self.tokens.tanh().double()) * (self.scale*gain).double()).to(x.dtype)


def indexed_graph(g, index, label=None):
    out = Data(x=torch.cat([g.x, torch.full((len(g.x), 1), float(index))], 1), edge_index=g.edge_index)
    if label is not None:
        out.y = label
    return out


class IndexedGraphEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.backbone = backbone

    def forward(self, x, edges, batch=None, prompt=None, prompt_type=None, gain=1):
        ids, x = x[:, -1].long(), x[:, :-1]
        if prompt is not None:
            x = prompt.inject(x, ids, gain)
        return self.backbone(x, edges, batch)


def class_means(embeddings, labels, classes):
    # Preserve ProG center_embedding's arithmetic and in-place class division.
    c = embeddings.new_zeros(classes, embeddings.size(1))
    c = c.scatter_add_(0, labels.unsqueeze(1).expand(-1, embeddings.size(1)), embeddings)
    counts = torch.bincount(labels, minlength=classes).unsqueeze(1).to(embeddings)
    for i in range(classes):
        if counts[i].item() != 0:
            c[i] /= counts[i]
    return c


@dataclass
class GraphFit:
    backbone: nn.Module
    prompt: GraphInputPrompt
    support: torch.Tensor
    support_labels: torch.Tensor
    config: dict
    trace: list
    classes: int
    audit: dict

    @torch.no_grad()
    def scores(self, graphs, query, *, batch_size=128):
        graphs = clean_graphs(graphs); device = self.prompt.tokens.device
        query = torch.as_tensor(query, dtype=torch.long).cpu()
        if query.ndim != 1 or not query.numel() or query.min() < 0 or query.max() >= len(graphs):
            raise ValueError('Nonempty valid query IDs required')
        enc = IndexedGraphEncoder(self.backbone); self.backbone.eval(); self.prompt.eval()
        def embed(ids):
            parts = []
            for b in DataLoader([indexed_graph(graphs[int(i)], int(i)) for i in ids], batch_size=batch_size):
                b = b.to(device)
                parts.append(enc(b.x, b.edge_index, b.batch, self.prompt, gain=self.config['inference_gain']).cpu())
            return torch.cat(parts)
        zs, zq = embed(self.support), embed(query)
        centers = torch.stack([zs[self.support_labels == c].mean(0) for c in range(self.classes)])
        return F.cosine_similarity(zq[:, None], centers[None], dim=-1)

    def state_dict(self):
        cpu = lambda module: {k:v.detach().cpu() for k,v in module.state_dict().items()}
        return dict(backbone=cpu(self.backbone), prompt=cpu(self.prompt), support=self.support,
                    support_labels=self.support_labels, config=self.config, trace=self.trace,
                    classes=self.classes, audit=self.audit)

    @classmethod
    def from_state_dict(cls, backbone, state, device='cpu'):
        backbone = copy.deepcopy(backbone).to(device).eval().requires_grad_(False)
        backbone.load_state_dict(state['backbone'], strict=True)
        sd = state['prompt']; prompt = GraphInputPrompt(sd['table'], sd['scale']).to(device)
        prompt.load_state_dict(sd, strict=True)
        return cls(backbone, prompt, state['support'].cpu(), state['support_labels'].cpu(),
                   state['config'], state['trace'], state['classes'], state['audit'])


def fit_graph(backbone, graphs, support, support_labels, classes, *, reference=None,
              seed=0, device='cpu', epochs=1000):
    """Fit one task. No query labels, query IDs or query-based early stopping.

    epochs is exposed for smoke checks; 1000 is the locked maximum. Other
    training settings are fixed: Adam .001/wd0, batch128, patience20, final state.
    """
    graphs = clean_graphs(graphs)
    si, sy = support_vectors(support, support_labels, classes, len(graphs), device='cpu')
    counts = torch.bincount(sy, minlength=classes)
    if not (counts == counts[0]).all() or int(counts[0]) not in [1, 3, 5]:
        raise ValueError('Locked graph protocol requires balanced 1/3/5-shot support')
    if not 1 <= epochs <= 1000:
        raise ValueError('epochs must be in [1,1000]')
    reference = reference or prepare_graph_reference(graphs)
    cfg = dict(reference.selection['shots'][str(int(counts[0]))]['full'])
    table, scale, _ = make_graph_routes(graphs, si, sy, classes, reference, cfg)
    model = copy.deepcopy(backbone).to(device).eval().requires_grad_(False); before = state_hash(model)
    seed_everything(seed)
    prompt = GraphInputPrompt(table, scale).to(device)
    opt = torch.optim.Adam(prompt.parameters(), lr=.001, weight_decay=0)
    enc = IndexedGraphEncoder(model)
    loader = DataLoader([indexed_graph(graphs[int(i)], int(i), sy[t]) for t,i in enumerate(si)], batch_size=128, shuffle=True)
    trace = []; best = 1e9; wait = 0
    for epoch in range(1, epochs+1):
        total = 0.; examples = 0
        for b in loader:
            opt.zero_grad(); b = b.to(device)
            z = enc(b.x, b.edge_index, b.batch, prompt)
            centers = class_means(z, b.y, classes)
            loss = F.cross_entropy(F.cosine_similarity(z.unsqueeze(1), centers.unsqueeze(0), dim=-1)/.1, b.y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite support loss')
            loss.backward()
            if prompt.tokens.grad is None or not torch.isfinite(prompt.tokens.grad).all(): raise RuntimeError('Invalid prompt gradient')
            grad = float(prompt.tokens.grad.norm()); opt.step()
            total += loss.item()*b.y.numel(); examples += b.y.numel()
        value = total/examples; trace.append(dict(epoch=epoch, loss=value, gradient_norm=grad))
        if value < best: best = value; wait = 0
        else:
            wait += 1
            if wait == 20: break
    if state_hash(model) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen backbone changed')
    audit = dict(backbone_frozen=True, prompt_reset=True, adam_reset=True, query_labels_received=False,
                 train_gain=1, final_state=True, prompt_parameters=prompt.tokens.numel(), steps=len(trace),
                 reference_scope='explicit unlabelled reference graph pool; transductive if it includes query')
    return GraphFit(model, prompt, si, sy, cfg, trace, classes, audit)
