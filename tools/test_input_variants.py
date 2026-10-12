"""CPU numerical checks against archived training and independent pooling loop.

Run from the repository root: python -m tools.test_input_variants
"""
import json
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import k_hop_subgraph

from prprompt.model import TwoLayerGCN, normalized_adjacency
from prprompt.input_variants import fit_fullgraph, fit_subgraph, SubgraphConfig
from experiments.f100_input_20261005.model import fit_input
from experiments.v1_f100.historical import load


def reference(model, x, edges, support, y, cache, steps):
    ns = load()
    torch.manual_seed(42)
    model.eval().requires_grad_(False)
    adj = normalized_adjacency(edges, len(x), x.dtype)
    r = torch.sparse.mm(adj, torch.sparse.mm(adj, x))
    raw = ns['rectified_teacher'](r-r.mean(0, keepdim=True), support, y, 2, 'rect_teacher_ref50', 42)[0]
    gate = ns['temperature_gate'](raw, adj, .1, 'raw')[0].detach()
    scale = x.std(0).clamp_min(1e-6)
    tokens = torch.nn.Parameter(torch.zeros(2, x.shape[1]))
    opt = torch.optim.Adam([tokens], lr=.01, weight_decay=0)
    def encode(ids):
        batch = Batch.from_data_list([Data(x=x[cache[i]['nodes']],
            edge_index=cache[i]['edges'], route=gate[cache[i]['nodes']]) for i in ids])
        prompted = batch.x + 10 * (batch.route @ tokens.tanh()) * scale
        return global_mean_pool(model(prompted, batch.edge_index), batch.batch)
    loo = bool(torch.bincount(y).min() > 1)
    for _ in range(steps):
        opt.zero_grad()
        z = encode(support.tolist())
        loss = F.cross_entropy(ns['logits'](z, torch.arange(len(y)), y, 2, loo=loo), y) + .001*tokens.square().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = encode(list(range(len(x))))
        scores = ns['logits'](z-z.mean(0, keepdim=True), support, y, 2)
    return scores, z, tokens, gate


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    x = torch.randn(24, 6)
    a = torch.arange(24)
    edges = torch.cat([torch.stack([a, (a+1)%24]), torch.stack([(a+1)%24, a])], 1)
    model = TwoLayerGCN(6, 8, 8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cache = {}
    for i in range(len(x)):
        nodes, local, _, _ = k_hop_subgraph(i, 2, edges, relabel_nodes=True)
        cache[i] = dict(nodes=nodes, edges=local)
    checks = []
    for support, labels in [(torch.tensor([0, 12]), torch.tensor([0, 1])),
                             (torch.tensor([0, 1, 12, 13]), torch.tensor([0, 0, 1, 1]))]:
        scores, z, _, _ = fit_fullgraph(model, x, edges, support, labels, 2, steps=4)
        original = fit_input('Soft', model, x, edges, support, labels, 2, 42, 4)
        torch.testing.assert_close(scores, original[0], rtol=0, atol=0)
        torch.testing.assert_close(z, original[1], rtol=0, atol=0)
        result = fit_subgraph(model, x, edges, support, labels, 2, cache.__getitem__,
                              config=SubgraphConfig(steps=4))
        ref = reference(model, x, edges, support, labels, cache, 4)
        for actual, expected in zip([result[0], result[1], result[2]['tokens'], result[2]['gate']], ref):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        checks.append(dict(support_per_class=len(labels)//2, fullgraph_exact=True,
                           subgraph_exact=True, backbone_unchanged=True))
    print(json.dumps(dict(device='cpu', torch_version=torch.__version__, checks=checks), indent=2))


if __name__ == '__main__':
    main()
