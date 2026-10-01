"""Every control and copied reference: pairing, hashes, independent metrics and replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prprompt.model import TwoLayerGCN, normalized_adjacency
from prprompt.protocol import sha_file, save_json, sample_support
from prprompt.train import tensor_hash
from experiments.v1_f100.historical import load
from experiments.boundary_20261002.model import saved_readout
from experiments.boundary_20261002.state_io import load_state
from experiments.f100_boundary_20261002.prepare import read_plan
from experiments.boundary_20261002.order_guard import verify_order_diagnostic
from experiments.f100_boundary_20261002.model import transform_gate
from experiments.f100_boundary_20261002.run import parent_methods, PAIR_FIELDS
from tools.metric_utils import numpy_metrics


def audit(a):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = read_plan(a.plan)
    job = plan['jobs'][a.job_index]
    spec = plan['datasets'][job['dataset']]
    assert sha_file(spec['snapshot']) == spec['snapshot_sha256']
    assert sha_file(job['checkpoint']) == job['checkpoint_sha256']
    data = torch.load(spec['snapshot'], weights_only=True, map_location='cpu')
    x, edges = data['x'].to(a.device), data['edges'].to(a.device)
    classes = spec['classes']
    model = TwoLayerGCN(x.shape[1]).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(job['checkpoint'], weights_only=True, map_location='cpu'))
    adj = normalized_adjacency(edges, len(x), x.dtype)
    old = load()
    with torch.no_grad():
        hidden = model.conv_layers[0](x, edges)
        attrs = torch.sparse.mm(adj, torch.sparse.mm(adj, x))
    splits = plan['split_seeds'][:1] if a.smoke else plan['split_seeds']
    seeds = plan['train_seeds'][:1] if a.smoke else plan['train_seeds']
    expected = {(s, t, 'F100_'+k) for s in splits for t in seeds for k in plan['controls']}
    root = a.plan.parent / ('smoke' if a.smoke else 'results') / job['id'] / str(a.shots)
    seen, refs_seen, details, cache = set(), set(), [], {}
    maximum_metric_error = 0.
    for p in sorted(root.glob('*/*/*/metrics.json')):
        r = json.loads(p.read_text())
        unit = (r['split_seed'], r['train_seed'], r['method'])
        assert unit in expected and unit not in seen
        seen.add(unit)
        assert r['plan_sha256'] == plan['plan_sha256'] and r['parent_plan_sha256'] == plan['parent_plan_sha256']
        assert r['smoke'] == a.smoke and r['family'] == 'Edgepred_Gprompt'
        f = p.parent
        for name, key in [('state.pt.gz', 'state_sha256'), ('predictions.npz', 'predictions_sha256'), ('selection.json', 'selection_sha256')]:
            assert sha_file(f / name) == r[key]
        seal = json.loads((f / 'seal.json').read_text())
        for k in ['plan_sha256', 'signature', 'support_sha256', 'query_sha256', 'predictions_sha256']:
            assert seal[k] == r[k]
        assert not json.loads((f / 'selection.json').read_text())['query_labels_used']
        state = load_state(f / 'state.pt.gz')
        s, y, q = (state[k].to(a.device) for k in ['support', 'support_labels', 'query'])
        assert torch.equal(state['support'], sample_support(data['pool'], data['labels'][data['pool']], classes, a.shots, r['split_seed']))
        assert torch.equal(data['labels'][state['support']], state['support_labels'])
        assert torch.equal(data['query'], state['query']) and not torch.isin(state['support'], state['query']).any()
        for field, key in [('support', 'support_sha256'), ('support_labels', 'support_labels_sha256'), ('query', 'query_sha256')]:
            assert tensor_hash(state[field]) == r[key]
        assert tensor_hash(data['labels'][state['query']]) == r['evaluation_labels_sha256']
        with np.load(f / 'predictions.npz') as saved:
            probs = saved['probabilities'].copy()
            assert np.array_equal(saved['support'], state['support'].numpy()) and np.array_equal(saved['query'], state['query'].numpy())
        metrics = numpy_metrics(data['labels'][state['query']].numpy(), probs)
        error = max(abs(metrics[k]-r['metrics'][k]) for k in metrics)
        assert error < 1e-12
        maximum_metric_error = max(maximum_metric_error, error)
        with torch.no_grad():
            z = state['raw_embedding'].to(a.device)
            assert tensor_hash(z) == r['embedding_sha256']
            verify_order_diagnostic(f, state, r['audit'])
            rebuilt = saved_readout(state, z, s, y, classes, a.device)[q].cpu().double().softmax(-1).numpy()
            assert np.array_equal(probs, rebuilt), ('saved readout drift', str(f))
            tokens, gate = state['prompt']['tokens'].to(a.device), state['prompt']['gate'].to(a.device)
            fresh = model.conv_layers[1](old['act'](hidden+10.*(gate@tokens.tanh())*hidden.std(0).clamp_min(1e-6)), edges)
            key = (r['support_sha256'], r['train_seed'])
            if key not in cache:
                raw = old['rectified_teacher'](attrs-attrs.mean(0, keepdim=True), s, y, classes, 'rect_teacher_ref50', r['train_seed'])[0]
                cache[key] = old['temperature_gate'](raw, adj, .1, 'raw')
            soft, info = cache[key]
            new_gate, degenerate = transform_gate(soft, info, adj, r['control'])
            gate_error = float((gate-new_gate).abs().max())
            fresh_full = model.conv_layers[1](old['act'](hidden+10.*(new_gate@tokens.tanh())*hidden.std(0).clamp_min(1e-6)), edges)
            fresh_prob = saved_readout(state, fresh_full, s, y, classes, a.device)[q].cpu().double().softmax(-1).numpy()
            raw_error = float((fresh-z).abs().max())
            mixed = float(((fresh-z).abs()/(2e-6+1e-6*z.abs())).max())
        assert set(r['references']) == set(parent_methods(a.shots))
        for method, refspec in r['references'].items():
            rf = a.plan.parent / refspec['relative']
            for name, expected_sha in refspec['files_sha256'].items(): assert sha_file(rf / name) == expected_sha
            ref = json.loads((rf / 'metrics.json').read_text())
            assert ref['plan_sha256'] == plan['parent_plan_sha256'] and ref['smoke'] == a.smoke
            assert ref['method'] == method
            assert all(ref[k] == r[k] for k in ['checkpoint_id', 'shots', 'split_seed', 'train_seed', 'split_policy_sha256', 'evaluation_labels_sha256'])
            assert sha_file(rf / 'predictions.npz') == ref['predictions_sha256']
            assert sha_file(rf / 'selection.json') == ref['selection_sha256']
            assert all(ref[k] == r[k] for k in PAIR_FIELDS)
            refseal = json.loads((rf / 'seal.json').read_text())
            assert refseal['signature'] == ref['signature'] and refseal['predictions_sha256'] == ref['predictions_sha256']
            refkey = (r['split_seed'], r['train_seed'], method)
            if refkey not in refs_seen:
                with np.load(rf / 'predictions.npz') as rp:
                    assert np.array_equal(rp['support'], state['support'].numpy()) and np.array_equal(rp['query'], state['query'].numpy())
                    rm = numpy_metrics(data['labels'][state['query']].numpy(), rp['probabilities'])
                re = max(abs(rm[k]-ref['metrics'][k]) for k in rm)
                assert re < 1e-12
                maximum_metric_error = max(maximum_metric_error, re)
                refs_seen.add(refkey)
        details.append(dict(method=r['method'], split_seed=r['split_seed'], train_seed=r['train_seed'],
            saved_readout_exact=True, metric_error=error, gate_max_error=gate_error, fixed_route_raw_error=raw_error,
            mixed_forward_ratio=mixed, predeclared_gate_forward_passed=gate_error < 2e-6 and mixed < 1,
            fresh_full_probability_error=float(np.max(abs(fresh_prob-probs))),
            fresh_full_class_changes=int(np.count_nonzero(fresh_prob.argmax(1) != probs.argmax(1))),
            hard_route_degenerate_saved=r['audit']['hard_route_degenerate'], hard_route_degenerate_fresh=degenerate,
            norm_match_feasible=r['audit']['norm_match_feasible'], soft_norm=r['audit']['soft_norm'], actual_norm=r['audit']['actual_norm'],
            pre_fit_filter_assert_failures=r['audit']['pre_fit_filter_assert_failures'],
            prefit_order_guard=r['audit']['prefit_order_guard']))
    assert seen == expected
    assert len(refs_seen) == len(splits)*len(seeds)*2
    summary = dict(plan_sha256=plan['plan_sha256'], checkpoint_id=job['id'], shots=a.shots, smoke=a.smoke,
        control_units=len(details), reference_units=len(refs_seen), maximum_metric_error=maximum_metric_error,
        hashes_pairing_and_saved_readout_passed=True,
        predeclared_gate_forward_failures=sum(not d['predeclared_gate_forward_passed'] for d in details),
        fresh_full_class_changes=sum(d['fresh_full_class_changes'] for d in details),
        hardmatched_degenerate=sum(d['method'] == 'F100_HardMatched' and d['hard_route_degenerate_saved'] for d in details),
        pre_fit_filter_assert_failure_count=sum(len(d['pre_fit_filter_assert_failures']) for d in details), details=details)
    summary.update(original_argmax_assertion_failed_units=sum(not d['prefit_order_guard']['original_argmax_assertion_passed'] for d in details),
        accepted_exact_tie_rows=sum(d['prefit_order_guard']['accepted_exact_tie_rows'] for d in details))
    output = a.plan.parent / ('smoke_analysis' if a.smoke else 'analysis') / (job['id']+'_'+str(a.shots)+'_audit.json')
    save_json(output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'details'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--job-index', type=int, required=True)
    p.add_argument('--shots', type=int, choices=[1, 3, 5], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke', action='store_true')
    audit(p.parse_args())
