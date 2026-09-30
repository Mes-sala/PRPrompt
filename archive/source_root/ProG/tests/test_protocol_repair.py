"""Behavioral regression tests for leakage, real prompt updates and budgets."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from prompt_graph.model import build_gnn
from prompt_graph.prompt.EdgePrompt import EdgePrompt, EdgePromptplus
from prompt_graph.tasker.audit import state_hash
from prompt_graph.tasker.strategies.edge_prompt import EdgePromptStrategy
from prompt_graph.tasker.strategies.self_pro import SelfProStrategy
from prompt_graph.tasker.strategy import TaskContext


def small_graph():
    n = 16
    ids = torch.arange(n)
    edges = torch.stack([torch.cat([ids, ids.roll(1)]), torch.cat([ids.roll(1), ids])])
    return Data(x=torch.randn(n, 6), edge_index=edges, y=ids % 2,
                train_mask=ids < 8, test_mask=ids >= 8)


@pytest.mark.parametrize('name,cls', [('EdgePrompt', EdgePrompt), ('EdgePromptplus', EdgePromptplus)])
def test_edge_prompt_updates_and_frozen_encoder(name, cls):
    data = small_graph()
    gnn = build_gnn('GCN', 6, 8, 2)
    ctx = TaskContext(gnn=gnn, prompt=cls([6, 8]), answering=nn.Linear(8, 2),
                      device=torch.device('cpu'), criterion=nn.CrossEntropyLoss(),
                      output_dim=2, extra={'prompt_type': name, 'lr': 0.01})
    strategy = EdgePromptStrategy()
    strategy.configure_optimizer(ctx)
    before_gnn = state_hash(gnn)
    before = [p.detach().clone() for p in ctx.prompt.parameters()]
    strategy.train_epoch(ctx, (data, torch.arange(8)))
    assert state_hash(gnn) == before_gnn
    assert not gnn.training
    assert any(not torch.equal(a, b) for a, b in zip(before, ctx.prompt.parameters()))
    for p in ctx.prompt.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    with torch.no_grad():
        expected = ctx.answering(gnn(data.x, data.edge_index, prompt=ctx.prompt, prompt_type=name))
    from prompt_graph.evaluation.node_scores import classification_scores
    assert strategy.evaluate(ctx, (data, torch.arange(8, 16))) == classification_scores(expected[8:], data.y[8:])


def test_selfpro_predictions_independent_of_query_labels():
    data = small_graph()
    ctx = TaskContext(gnn=build_gnn('GCN', 6, 8, 2), prompt=nn.Linear(8, 8),
                      device=torch.device('cpu'), output_dim=2)
    s = SelfProStrategy()
    query, support = torch.arange(8, 16), torch.arange(8)
    before = s.predict_logits(ctx, data, query, support, data.y[support].clone())
    data.y[query] = 1 - data.y[query]
    after = s.predict_logits(ctx, data, query, support, data.y[support].clone())
    torch.testing.assert_close(before, after)
    with pytest.raises(ValueError, match='requires support'):
        s.evaluate(ctx, (data, query))


def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_protocol_cell.py'
    spec = importlib.util.spec_from_file_location('protocol_runner', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config(tmp_path, prompt):
    return dict(dataset='Cora', prompt=prompt, checkpoint='None', gnn='GCN', hidden_dim=8,
                layers=2, shot=2, folds=2, support_passes=2, learning_rate=0.01,
                weight_decay=0., split_seed=13, train_seed=7, device='cpu',
                output_root=str(tmp_path/'results'), split_root=str(tmp_path/'splits'),
                subgraph_min=2, subgraph_max=5, allow_unverified=True,
                answer_passes=1, prompt_passes=1, prompt_lr=0.01,
                prompt_options={'supt_clusters': 2})


@pytest.mark.parametrize('prompt', ['None', 'Frozen', 'FrozenGraph', 'FrozenGraphPrototype',
    'GPF', 'GPF-plus', 'Gprompt', 'All-in-one', 'SUPT-soft', 'SUPT-hard', 'GraphTOP',
    'SelfPro', 'PSP', 'EdgePrompt', 'EdgePromptplus', 'GPPT',
    'FrozenNodePrototype', 'FrozenIdentityPrototype', 'GraphTOP-Frozen'])
def test_fixed_budget_cell(tmp_path, prompt):
    data = small_graph()
    cfg = config(tmp_path, prompt)
    result = runner().run(cfg, data=data)
    assert result['status'] == 'complete'
    assert len(result['folds']) == 2
    for fold in result['folds']:
        steps = sum(fold['optimizer_steps'].values())
        assert steps == (0 if prompt in {'FrozenGraphPrototype', 'FrozenNodePrototype', 'FrozenIdentityPrototype'} else 2)
        assert fold['support_passes'] == (0 if prompt in {'FrozenGraphPrototype', 'FrozenNodePrototype', 'FrozenIdentityPrototype'} else 2)
        if prompt != 'None':
            assert fold['encoder_hash_before'] == fold['encoder_hash_after']
        else:
            assert fold['parameters']['gnn']['changed_elements'] > 0
        assert fold['support_count'] == 4
        assert fold['query_count'] == 8


def test_config_validation_and_identity(tmp_path):
    mod = runner()
    cfg = config(tmp_path, 'Frozen')
    mod.validate(cfg)
    with pytest.raises(ValueError, match='unknown fields'):
        mod.validate(cfg | {'batch_size': 32})
    with pytest.raises(NotImplementedError):
        mod.validate(cfg | {'prompt': 'GraphPrompter'})
    data = small_graph()
    first = mod.run(cfg, data=data)
    again = mod.run(cfg, data=data)
    assert first['run_id'] == again['run_id']
    changed = mod.run(cfg | {'learning_rate': 0.02}, data=data)
    assert first['run_id'] != changed['run_id']
    # A different training seed must reuse identical support assets.
    reseeded = mod.run(cfg | {'train_seed': 9}, data=data)
    assert first['split_hashes'] == reseeded['split_hashes']


@pytest.mark.parametrize('prompt', ['None', 'GPF-plus', 'Gprompt', 'SelfPro', 'PSP',
                                    'EdgePrompt', 'GraphTOP', 'FrozenNodePrototype'])
def test_query_label_permutation_does_not_change_predictions(tmp_path, prompt):
    import json
    mod = runner()
    cfg = config(tmp_path, prompt) | {'folds': 1}
    data = small_graph()
    a = mod.run(cfg, data=data)
    data.y[data.test_mask] = 1 - data.y[data.test_mask]
    b = mod.run(cfg, data=data)
    pa = json.loads((Path(a['output_directory']) / 'predictions.json').read_text())[0]
    pb = json.loads((Path(b['output_directory']) / 'predictions.json').read_text())[0]
    torch.testing.assert_close(torch.tensor(pa['probabilities']), torch.tensor(pb['probabilities']))


def test_constant_loss_does_not_trigger_legacy_early_stop(tmp_path):
    cfg = config(tmp_path, 'Frozen') | {'support_passes': 25, 'folds': 1, 'learning_rate': 1e-30}
    result = runner().run(cfg, data=small_graph())
    assert result['folds'][0]['optimizer_steps']['optimizer'] == 25


def test_tampered_prediction_cache_is_rejected(tmp_path):
    mod = runner()
    cfg = config(tmp_path, 'Frozen')
    data = small_graph()
    result = mod.run(cfg, data=data)
    (Path(result['output_directory']) / 'predictions.json').write_text('[]')
    with pytest.raises(RuntimeError, match='predictions modified'):
        mod.run(cfg, data=data)
