"""RSP mathematical constraints and real protocol boundary regressions."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from prompt_graph.model import build_gnn
from prompt_graph.prompt.RSP import (RSPOptions, budget_edges, cached_bank, encode_bank,
    graph_context, inner_folds, mutual_neighbors, prototype_logits, select_prompt)
from prompt_graph.tasker.audit import state_hash


def graph():
    # Local edges mostly cross classes; same-class features permit remote retrieval.
    ids = torch.arange(40)
    edges = torch.stack([torch.cat([ids, ids.roll(1)]), torch.cat([ids.roll(1), ids])])
    generator = torch.Generator().manual_seed(19)
    x = F.one_hot(ids % 2, 2).float().repeat(1, 3) + .02 * torch.rand(40, 6, generator=generator)
    return Data(x=x, edge_index=edges, y=ids % 2, train_mask=ids < 20, test_mask=ids >= 20)


def encoder():
    with torch.random.fork_rng():
        torch.manual_seed(8)
        return build_gnn('GCN', 6, 8, 2).eval().requires_grad_(False)


def test_candidates_exclude_two_hops_and_preserve_budget():
    data = graph()
    _, _, degree, neighbors = graph_context(data.x, data.edge_index)
    edges, weights = mutual_neighbors(data.x, neighbors, 3, 7)
    other = mutual_neighbors(data.x, neighbors, 3, 40)
    torch.testing.assert_close(edges, other[0])
    torch.testing.assert_close(weights, other[1])
    assert edges.numel() > 0
    directed = set(map(tuple, edges.t().tolist()))
    for i, j in directed:
        assert (j, i) in directed and j != i and j not in neighbors[i]
        assert all(j not in neighbors[k] for k in neighbors[i])
    bounded = budget_edges(edges, weights, degree)
    mass = torch.zeros_like(degree).scatter_add_(0, edges[0], bounded)
    assert (mass <= degree + 1e-6).all()


def test_ties_zero_features_and_isolated_nodes():
    neighbors = [set() for _ in range(4)]
    edges, _ = mutual_neighbors(torch.ones(4, 2), neighbors, 1, 2)
    assert edges.tolist() == [[0, 1], [1, 0]]
    edges, weights = mutual_neighbors(torch.zeros(4, 2), neighbors, 2, 2)
    assert edges.shape == (2, 0) and weights.numel() == 0


@pytest.mark.parametrize('mode', ['retrieval', 'local', 'random', 'self_loop', 'identity', 'raw_features'])
def test_bank_identity_freeze_and_real_weighted_path(mode):
    data, gnn = graph(), encoder()
    before = state_hash(gnn)
    seen_weights = []
    handles = [conv.register_forward_pre_hook(lambda module, args: seen_weights.append(
        args[2].clone() if len(args) > 2 and args[2] is not None else None)) for conv in gnn.conv_layers]
    bank = encode_bank(gnn, data.x, data.edge_index, RSPOptions.parse({'mode': mode, 'k': 3}))
    for handle in handles:
        handle.remove()
    assert before == state_hash(gnn)
    expected = data.x if mode == 'raw_features' else gnn(data.x, data.edge_index)
    torch.testing.assert_close(bank['embeddings'][0], expected, rtol=0, atol=0)
    assert all(v['max_relative_added_degree'] <= .200001 for v in bank['candidate_stats'])
    if mode not in {'identity', 'raw_features'}:
        assert bank['build_forward_count'] == 6
        assert len(seen_weights) == 12 and all(w is not None for w in seen_weights[2:])
    if mode == 'retrieval':
        assert not torch.equal(bank['embeddings'][0], bank['embeddings'][2])


def test_inner_validation_excludes_each_class_holdout():
    support, labels = torch.arange(10), torch.arange(10) % 2
    assignment = inner_folds(support, labels, 2, 5, 11)
    for f in range(5):
        held = assignment == f
        assert torch.bincount(labels[held]).tolist() == [1, 1]
        assert torch.bincount(labels[~held]).tolist() == [4, 4]
        z = F.normalize(torch.arange(30).float().view(10, 3) + 1, dim=1)
        result = prototype_logits(z, support[~held], labels[~held], support[held], 2, .1)
        centers = torch.stack([z[support[(~held) & (labels == c)]].mean(0) for c in range(2)])
        torch.testing.assert_close(result, z[support[held]] @ F.normalize(centers, dim=1).t() / .1)
    with pytest.raises(ValueError, match='divisible'):
        inner_folds(torch.arange(2), torch.arange(2), 2, 5, 11)


def test_selection_tie_and_query_labels_cannot_enter():
    data, gnn = graph(), encoder()
    opts = RSPOptions.parse()
    bank = encode_bank(gnn, data.x, data.edge_index, opts)
    support = torch.arange(10)
    selected, record = select_prompt(bank, support, data.y[support], 2, opts)
    data.y[20:] = 1 - data.y[20:]
    again, record2 = select_prompt(bank, support, data.y[support], 2, opts)
    assert record == record2
    torch.testing.assert_close(selected, again)
    same = dict(bank, embeddings=bank['embeddings'][0:1].repeat(6, 1, 1))
    _, zero = select_prompt(same, support, data.y[support], 2, opts)
    assert zero['selected_index'] == 0


def test_cache_reuse_invalidation_and_corruption(tmp_path):
    data, gnn = graph(), encoder()
    opts = RSPOptions.parse({'cache_root': str(tmp_path)})
    bank, first = cached_bank(gnn, data.x, data.edge_index, opts)
    other, second = cached_bank(gnn, data.x, data.edge_index, opts)
    assert not first['cache_hit'] and second['cache_hit']
    assert second['encoder_forward_count'] == 0
    torch.testing.assert_close(bank['embeddings'], other['embeddings'])
    with torch.no_grad():
        next(gnn.parameters()).add_(.01)
    _, third = cached_bank(gnn, data.x, data.edge_index, opts)
    assert third['cache_key'] != first['cache_key']
    bank['embeddings'][0, 0, 0] += 1
    torch.save(bank, tmp_path / f'{first["cache_key"]}.pt')
    with pytest.raises(RuntimeError, match='integrity'):
        cached_bank(encoder(), data.x, data.edge_index, opts)


def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_protocol_cell.py'
    spec = importlib.util.spec_from_file_location('rsp_test_runner', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config(tmp_path):
    return dict(dataset='Cora', prompt='RSP', checkpoint='None', gnn='GCN', hidden_dim=8,
                layers=2, shot=5, folds=2, support_passes=0, learning_rate=0., weight_decay=0.,
                split_seed=13, train_seed=7, device='cpu', output_root=str(tmp_path / 'results'),
                split_root=str(tmp_path / 'splits'), prompt_options={'cache_root': str(tmp_path / 'cache')})


def test_protocol_end_to_end_and_resume(tmp_path):
    mod, cfg = runner(), config(tmp_path)
    result = mod.run(cfg, data=graph())
    assert result['status'] == 'complete' and result['selection'] == 'support_inner_cv'
    folds = result['folds']
    assert folds[0]['encoder_forward_count'] == 6 and folds[1]['encoder_forward_count'] == 0
    assert folds[0]['encoder_hash_before'] == folds[1]['encoder_hash_before']
    for fold in folds:
        assert fold['completed_epochs'] == fold['support_passes'] == fold['backward_count'] == 0
        assert fold['optimizer_steps'] == {}
        assert fold['encoder_hash_before'] == fold['encoder_hash_after']
        assert fold['support_count'] == 10 and fold['query_count'] == 20
    assert mod.run(cfg, data=graph())['run_id'] == result['run_id']
    different = mod.run(cfg | {'split_seed': 14}, data=graph())
    assert different['folds'][0]['cache_hit']
    assert different['folds'][0]['cache_key'] == folds[0]['cache_key']
    # Exercise the actual entry point with different query labels, not just a helper.
    changed = graph()
    changed.y[changed.test_mask] = 1 - changed.y[changed.test_mask]
    relabeled = mod.run(cfg, data=changed)
    import json
    first_preds = json.loads((Path(result['output_directory']) / 'predictions.json').read_text())
    second_preds = json.loads((Path(relabeled['output_directory']) / 'predictions.json').read_text())
    assert [p['probabilities'] for p in first_preds] == [p['probabilities'] for p in second_preds]
    assert relabeled['folds'][0]['selected_index'] == folds[0]['selected_index']


def test_cached_encoder_must_still_be_frozen(tmp_path):
    data, gnn = graph(), encoder()
    opts = RSPOptions.parse({'cache_root': str(tmp_path)})
    cached_bank(gnn, data.x, data.edge_index, opts)
    gnn.train()
    with pytest.raises(ValueError, match='eval'):
        cached_bank(gnn, data.x, data.edge_index, opts)


@pytest.mark.parametrize('change', [{'support_passes': 200}, {'learning_rate': .01}, {'shot': 1},
    {'prompt_options': {'temperature': 0}}, {'prompt_options': {'typo': 1}}])
def test_reject_invalid_protocol(tmp_path, change):
    with pytest.raises((ValueError, TypeError)):
        runner().validate(config(tmp_path) | change)


def test_reject_directed_and_duplicate_input():
    data = graph()
    with pytest.raises(ValueError, match='undirected'):
        graph_context(data.x, data.edge_index[:, 1:])
    with pytest.raises(ValueError, match='Duplicate'):
        graph_context(data.x, torch.cat([data.edge_index, data.edge_index[:, :1]], dim=1))
