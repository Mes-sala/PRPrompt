"""Label-free retrieval topology bank and support-only discrete adaptation.

No optimizer, pseudo-labels, or trainable encoder parameters. Exact retrieval
uses CPU blocks, with deterministic node-ID tie breaking.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import add_remaining_self_loops, coalesce

from prompt_graph.tasker.audit import state_hash


DEFAULT_CANDIDATES = ((0., 0.), (.05, 0.), (.20, 0.),
                      (0., .05), (0., .20), (.10, .10))


@dataclass(frozen=True)
class RSPOptions:
    k: int = 5
    block_size: int = 256
    temperature: float = .1
    penalty: float = .05
    inner_split_seed: int = 2718
    inner_folds: int = 5
    candidates: tuple = DEFAULT_CANDIDATES
    mode: str = 'retrieval'
    control_seed: int = 31415
    fixed_index: int | None = None
    cache_root: str | None = None

    @classmethod
    def parse(cls, options=None):
        values = dict(options or {})
        if 'candidates' in values:
            values['candidates'] = tuple(tuple(p) for p in values['candidates'])
        obj = cls(**values)  # unknown fields must fail, not silently default
        for key in ('k', 'block_size', 'inner_folds'):
            if type(getattr(obj, key)) is not int or getattr(obj, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if obj.inner_folds < 2:
            raise ValueError('inner_folds must be >=2')
        for key in ('inner_split_seed', 'control_seed'):
            if type(getattr(obj, key)) is not int or getattr(obj, key) < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if not math.isfinite(obj.temperature) or obj.temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if not math.isfinite(obj.penalty) or obj.penalty < 0:
            raise ValueError('penalty must be finite and nonnegative')
        if not obj.candidates or obj.candidates[0] != (0., 0.):
            raise ValueError('First candidate must be exact identity (0,0)')
        for pair in obj.candidates:
            if len(pair) != 2 or any(not math.isfinite(v) or v < 0 for v in pair) or sum(pair) > .2 + 1e-12:
                raise ValueError('Candidates require two nonnegative amplitudes with sum <=0.2')
        if len(set(obj.candidates)) != len(obj.candidates):
            raise ValueError('Duplicate candidates')
        if obj.mode not in {'retrieval', 'local', 'random', 'self_loop', 'identity', 'raw_features'}:
            raise ValueError('Unknown RSP mode')
        if obj.fixed_index is not None and (type(obj.fixed_index) is not int or not 0 <= obj.fixed_index < len(obj.candidates)):
            raise ValueError('fixed_index outside candidate bank')
        if obj.mode in {'identity', 'raw_features'} and obj.fixed_index not in (None, 0):
            raise ValueError('Baseline modes only allow index 0')
        return obj


def tensor_hash(*tensors):
    digest = hashlib.sha256()
    for value in tensors:
        value = value.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def graph_context(x, edge_index):
    """Validate original unweighted undirected graph; match GCN self loops."""
    x, edge_index = x.detach().cpu(), edge_index.detach().cpu()
    n = x.size(0)
    if x.ndim != 2 or n == 0 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('RSP requires finite floating node features')
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError('Invalid edge_index')
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n):
        raise ValueError('Edge index outside graph')
    ids = edge_index[0] * n + edge_index[1]
    if ids.unique().numel() != ids.numel():
        raise ValueError('Duplicate original edges must be audited before RSP')
    if not torch.equal(ids.sort().values, (edge_index[1] * n + edge_index[0]).sort().values):
        raise ValueError('RSP requires an undirected graph; no implicit symmetrization')
    base_edges, base_weights = add_remaining_self_loops(
        edge_index, torch.ones(edge_index.size(1), dtype=x.dtype), fill_value=1., num_nodes=n)
    degree = torch.zeros(n, dtype=x.dtype).scatter_add_(0, base_edges[0], base_weights)
    neighbors = [set() for _ in range(n)]
    for i, j in edge_index.t().tolist():
        if i != j:
            neighbors[i].add(j)
    return base_edges, base_weights, degree, neighbors


def mutual_neighbors(features, neighbors, k, block_size, local=False):
    """Positive mutual top-k outside two hops (or within two for control)."""
    z = F.normalize(features.detach().cpu(), dim=1)
    if not torch.isfinite(z).all():
        raise ValueError('Nonfinite retrieval representation')
    n = z.size(0)
    directed = {}
    for start in range(0, n, block_size):
        scores = z[start:start + block_size] @ z.t()
        for offset in range(scores.size(0)):
            i = start + offset
            near = {i} | neighbors[i]
            for j in neighbors[i]:
                near.update(neighbors[j])
            if local:
                allowed = near - neighbors[i] - {i}
                mask = torch.ones(n, dtype=torch.bool)
                if allowed:
                    mask[list(allowed)] = False
                scores[offset, mask] = -torch.inf
            else:
                scores[offset, list(near)] = -torch.inf
            # Stable descending sort retains ascending node IDs in ties.
            for j in torch.argsort(scores[offset], descending=True, stable=True)[:k].tolist():
                value = float(scores[offset, j])
                if value > 0:
                    directed[(i, j)] = value
    pairs = [(i, j) for i, j in directed if i < j and (j, i) in directed]
    pairs.sort()
    entries, weights = [], []
    for i, j in pairs:
        entries.extend([(i, j), (j, i)])
        # Symmetric even if device BLAS rounding differs for transposed entries.
        value = (directed[i, j] + directed[j, i]) / 2
        weights.extend([value, value])
    edges = torch.tensor(entries, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return edges, torch.tensor(weights, dtype=features.dtype)


def budget_edges(edges, weights, degree):
    q = torch.zeros_like(degree).scatter_add_(0, edges[0], weights)
    scale = degree / q.clamp_min(torch.finfo(q.dtype).tiny)
    return weights * torch.minimum(scale[edges[0]], scale[edges[1]])


def randomize_edges(edges, weights, neighbors, seed):
    """Permute endpoints, discard invalid/local edges, then reapply budget.

    This is not an exactly degree-matched null. Retained counts are recorded.
    """
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(neighbors), generator=generator)
    mapped = perm[edges]
    keep = []
    for i, j in mapped.t().tolist():
        near = {i} | neighbors[i]
        for v in neighbors[i]:
            near.update(neighbors[v])
        keep.append(j not in near)
    keep = torch.tensor(keep, dtype=torch.bool)
    return mapped[:, keep], weights[keep]


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def validate_encoder(gnn):
    if getattr(gnn, 'conv_type', None) != 'GCN':
        raise ValueError('RSP MVP supports ProG GCN only')
    if any(getattr(c, 'cached', False) or not getattr(c, 'normalize', False)
           or not getattr(c, 'add_self_loops', False) for c in gnn.conv_layers):
        raise ValueError('RSP requires uncached normalized GCN with self loops')
    if gnn.training or any(p.requires_grad for p in gnn.parameters()):
        raise ValueError('Freeze encoder and set eval() before RSP')


@torch.no_grad()
def encode_bank(gnn, x, edge_index, options):
    """Only X, A and encoder enter this boundary: labels cannot enter cache."""
    validate_encoder(gnn)
    device = next(gnn.parameters()).device
    _sync(device)
    start = time.perf_counter()
    before = state_hash(gnn)
    base, bw, degree, neighbors = graph_context(x, edge_index)
    x, edge_index = x.to(device), edge_index.to(device)
    if options.mode == 'raw_features':
        h0, forward_count = x.detach().cpu(), 0
    else:
        h0, forward_count = gnn(x, edge_index).detach().cpu(), 1
    if not torch.isfinite(h0).all():
        raise FloatingPointError('Nonfinite encoder output')
    embeddings, diagnostics, graphs = [h0], [], []
    retrieval_start = time.perf_counter()
    if options.mode not in {'identity', 'raw_features'}:
        for m, features in enumerate((x.cpu(), h0)):
            edges, weights = mutual_neighbors(features, neighbors, options.k,
                                               options.block_size, options.mode == 'local')
            retrieved_count = edges.size(1)
            if options.mode == 'random':
                edges, weights = randomize_edges(edges, weights, neighbors, options.control_seed + m)
            weights = budget_edges(edges, weights, degree)
            mass = torch.zeros_like(degree).scatter_add_(0, edges[0], weights)
            graphs.append((edges, weights))
            diagnostics.append(dict(retrieved_directed_edges=retrieved_count,
                                    directed_edges=edges.size(1),
                                    graph_hash=tensor_hash(edges, weights),
                                    degree_mass=mass.tolist()))
    retrieval_seconds = time.perf_counter() - retrieval_start
    candidates = ((0., 0.),) if options.mode in {'identity', 'raw_features'} else options.candidates
    candidate_stats = [dict(alpha=0., beta=0., added_directed_edges=0,
                           max_relative_added_degree=0.)]
    for alpha, beta in candidates[1:]:
        extra_e = torch.cat([graphs[0][0], graphs[1][0]], dim=1)
        extra_w = torch.cat([alpha * graphs[0][1], beta * graphs[1][1]])
        positive = extra_w > 0
        extra_e, extra_w = extra_e[:, positive], extra_w[positive]
        added_degree = torch.zeros_like(degree).scatter_add_(0, extra_e[0], extra_w)
        if options.mode == 'self_loop':
            nodes = torch.arange(degree.numel())
            extra_e, extra_w = torch.stack([nodes, nodes]), added_degree
        edges, weights = coalesce(torch.cat([base, extra_e], dim=1), torch.cat([bw, extra_w]),
                                  num_nodes=degree.numel(), reduce='sum')
        h = gnn(x, edges.to(device), edge_weight=weights.to(device)).detach().cpu()
        if not torch.isfinite(h).all():
            raise FloatingPointError('Nonfinite prompted representation')
        embeddings.append(h)
        forward_count += 1
        candidate_stats.append(dict(alpha=alpha, beta=beta,
            added_directed_edges=edges.size(1) - base.size(1),
            max_relative_added_degree=float((added_degree / degree).max()),
            graph_hash=tensor_hash(edges, weights)))
    if state_hash(gnn) != before:
        raise RuntimeError('Encoder state changed while encoding RSP bank')
    _sync(device)
    return dict(embeddings=torch.stack(embeddings), candidates=list(candidates),
                retrieval=diagnostics, candidate_stats=candidate_stats,
                build_forward_count=forward_count, retrieval_seconds=retrieval_seconds,
                build_seconds=time.perf_counter() - start)


def cached_bank(gnn, x, edges, options):
    """Content-addressed, locked, atomic cache excluding labels and split IDs."""
    validate_encoder(gnn)
    graph_options = {key: getattr(options, key) for key in
                     ('k', 'block_size', 'candidates', 'mode', 'control_seed')}
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        source.update(str(path.relative_to(root)).encode())
        source.update(path.read_bytes())
    identity = dict(version=1, graph=tensor_hash(x, edges), encoder=state_hash(gnn),
                    options=graph_options, source=source.hexdigest(),
                    torch=str(torch.__version__), pyg=str(torch_geometric.__version__),
                    platform=platform.platform(), device=str(next(gnn.parameters()).device),
                    threads=torch.get_num_threads(), encoder_repr=repr(gnn),
                    jk=gnn.JK, drop_ratio=gnn.drop_ratio)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    start = time.perf_counter()
    def build():
        bank = encode_bank(gnn, x, edges, options)
        bank['identity'] = identity
        bank['embedding_hash'] = tensor_hash(bank['embeddings'])
        return bank
    if options.cache_root is None:
        bank, hit = build(), False
    else:
        folder = Path(options.cache_root).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{key}.pt'
        with open(folder / f'{key}.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                bank = torch.load(path, map_location='cpu', weights_only=True)
                if bank['identity'] != identity or tensor_hash(bank['embeddings']) != bank['embedding_hash']:
                    raise RuntimeError('RSP cache integrity mismatch; inspect cache before rerunning')
                hit = True
            else:
                bank, hit = build(), False
                fd, temp = tempfile.mkstemp(prefix=f'{key}.', suffix='.tmp', dir=folder)
                os.close(fd)
                try:
                    torch.save(bank, temp)
                    os.replace(temp, path)
                finally:
                    Path(temp).unlink(missing_ok=True)
    return bank, dict(cache_key=key, cache_hit=hit,
                      cache_seconds=time.perf_counter() - start,
                      encoder_forward_count=0 if hit else bank['build_forward_count'],
                      embedding_bytes=bank['embeddings'].numel() * bank['embeddings'].element_size())


def prototype_logits(z, support, labels, query, classes, temperature):
    """z contains normalized node embeddings, prototypes exclude query by caller."""
    centers = []
    for c in range(classes):
        ids = support[labels == c]
        if not ids.numel():
            raise ValueError('Every class must have an anchor')
        centers.append(z[ids].mean(0))
    return z[query] @ F.normalize(torch.stack(centers), dim=1).t() / temperature


def inner_folds(support, labels, classes, count, seed):
    if support.ndim != 1 or labels.shape != support.shape or support.unique().numel() != support.numel():
        raise ValueError('Support indices must be unique with aligned labels')
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= classes:
        raise ValueError('Invalid support classes')
    groups = [(labels == c).nonzero().view(-1) for c in range(classes)]
    sizes = {len(g) for g in groups}
    if len(sizes) != 1 or min(sizes) < count or min(sizes) % count:
        raise ValueError('Balanced support count per class must be divisible by inner_folds and >= inner_folds')
    generator = torch.Generator().manual_seed(seed)
    assignment = torch.empty_like(labels)
    for group in groups:
        order = group[torch.randperm(len(group), generator=generator)]
        assignment[order] = torch.arange(len(group)) % count
    return assignment


@torch.no_grad()
def select_prompt(bank, support, labels, classes, options):
    support, labels = support.detach().cpu().long().view(-1), labels.detach().cpu().long().view(-1)
    z = F.normalize(bank['embeddings'], dim=2)
    assignment = inner_folds(support, labels, classes, options.inner_folds, options.inner_split_seed)
    losses = []
    for candidate in z:
        fold_losses = []
        for f in range(options.inner_folds):
            held = assignment == f
            logits = prototype_logits(candidate, support[~held], labels[~held], support[held], classes, options.temperature)
            fold_losses.append(float(F.cross_entropy(logits, labels[held])))
        losses.append(fold_losses)
    means = [sum(row) / len(row) for row in losses]
    penalized = [v + options.penalty * sum(pair) for v, pair in zip(means, bank['candidates'])]
    selected = (options.fixed_index if options.fixed_index is not None else
                min(range(len(means)), key=lambda i: (penalized[i], sum(bank['candidates'][i]), i)))
    return z[selected], dict(selected_index=selected, selected_amplitudes=bank['candidates'][selected],
                            inner_losses=losses, mean_inner_losses=means, penalized_losses=penalized,
                            selection='fixed_index' if options.fixed_index is not None else 'support_inner_cv',
                            inner_assignment=assignment.tolist(), inner_split_seed=options.inner_split_seed,
                            inner_split_hash=tensor_hash(support, labels, assignment))
