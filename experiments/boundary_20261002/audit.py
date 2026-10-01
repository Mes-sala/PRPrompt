from experiments.boundary_20261002.state_io import save_state, load_state
"""Every-unit hashes, independent scoring, saved readout and fresh-forward diagnostics."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from prprompt.model import TwoLayerGCN, normalized_adjacency
from prprompt.protocol import sha_file, save_json
from prprompt.train import tensor_hash
from tools.metric_utils import numpy_metrics
from experiments.v1_f100.historical import load
from experiments.boundary_20261002.prepare import read_plan
from experiments.boundary_20261002.order_guard import verify_order_diagnostic
from experiments.boundary_20261002.model import saved_readout, readout, routing, replay_baseline_embedding, BASELINES

def audit(args):
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plan = read_plan(args.plan); job = plan['jobs'][args.job_index]; shots = args.shots
    spec = plan['datasets'][job['dataset']]
    data = torch.load(spec['snapshot'], map_location='cpu', weights_only=True)
    x, edges = data['x'].to(args.device), data['edges'].to(args.device)
    classes = spec['classes']; device = args.device
    model = TwoLayerGCN(x.shape[1]).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(job['checkpoint'], map_location='cpu', weights_only=True))
    adj = normalized_adjacency(edges, len(x), x.dtype)
    with torch.no_grad():
        hidden = model.conv_layers[0](x, edges).detach(); base = model(x, edges).detach()
        raw_attrs = torch.sparse.mm(adj, torch.sparse.mm(adj, x)).detach()
    route_cache = {}; old = load(); details = []; metric_error = 0.; pairing = {}
    split_seeds = plan['split_seeds'][:1] if args.smoke else plan['split_seeds']
    train_seeds = plan['train_seeds'][:1] if args.smoke else plan['train_seeds']
    for probe_stage in [False, True]:
        names = [(p+'__'+h) for p in plan['readout_probe_sources'] for h in plan['probe_heads']] if probe_stage else plan['methods_by_shot'][str(shots)]
        dirname = ('smoke_probes' if args.smoke else 'probes') if probe_stage else ('smoke' if args.smoke else 'results')
        root = args.plan.parent/dirname/job['id']/str(shots)
        expected = {(s, t, m) for s in split_seeds for t in train_seeds for m in names}
        seen = set()
        for path in sorted(root.glob('*/*/*/metrics.json')):
            r = json.loads(path.read_text()); unit = (r['split_seed'], r['train_seed'], r['method'])
            assert unit in expected and unit not in seen; seen.add(unit)
            assert r['plan_sha256'] == plan['plan_sha256'] and r['smoke'] == args.smoke
            folder = path.parent
            for fn, k in [('state.pt.gz', 'state_sha256'), ('selection.json', 'selection_sha256'), ('predictions.npz', 'predictions_sha256')]:
                assert sha_file(folder/fn) == r[k], str(folder/fn)
            seal = json.loads((folder/'seal.json').read_text())
            for k in ['plan_sha256', 'signature', 'support_sha256', 'query_sha256', 'predictions_sha256']:
                assert seal[k] == r[k]
            selection = json.loads((folder/'selection.json').read_text()); assert not selection['query_labels_used']
            state = load_state(folder/'state.pt.gz')
            s, y, q = (state[k].to(device) for k in ['support', 'support_labels', 'query'])
            with np.load(folder/'predictions.npz') as saved:
                probabilities = saved['probabilities'].copy()
                assert np.array_equal(saved['support'], state['support'].numpy())
                assert np.array_equal(saved['query'], state['query'].numpy())
            assert not torch.isin(state['support'], state['query']).any()
            assert torch.equal(data['labels'][state['support']], state['support_labels'])
            assert tensor_hash(state['support']) == r['support_sha256']
            assert tensor_hash(state['query']) == r['query_sha256']
            pair = tuple(r[k] for k in ['checkpoint_sha256', 'graph_sha256', 'split_policy_sha256', 'support_sha256', 'query_sha256', 'support_labels_sha256'])
            assert r['split_seed'] not in pairing or pairing[r['split_seed']] == pair
            pairing[r['split_seed']] = pair
            metrics = numpy_metrics(data['labels'][state['query']].numpy(), probabilities)
            e = max(abs(metrics[k]-r['metrics'][k]) for k in metrics); assert e < 1e-12; metric_error = max(metric_error, e)
            with torch.no_grad():
                if probe_stage:
                    parent = args.plan.parent/state['parent_relative']
                    assert sha_file(parent/'state.pt.gz') == r['parent_state_sha256']
                    assert sha_file(parent/'metrics.json') == r['parent_metrics_sha256']
                    parent_state = load_state(parent/'state.pt.gz')
                    z = parent_state['raw_embedding'].to(device)
                    assert tensor_hash(z) == r['embedding_sha256']
                    head = None
                    if state['head_kind'] == 'Linear':
                        head = nn.Linear(z.shape[1], classes).to(device); head.load_state_dict(state['head'])
                    scores = readout(z, state['head_kind'], s, y, classes, head)
                else:
                    z = state['raw_embedding'].to(device)
                    assert tensor_hash(z) == r['embedding_sha256']
                    scores = saved_readout(state, z, s, y, classes, device)
                rebuilt = scores[q].cpu().double().softmax(-1).numpy()
                assert np.array_equal(probabilities, rebuilt), ('saved readout drift', str(folder), float(np.max(abs(probabilities-rebuilt))))
                detail = {'method': r['method'], 'shots': shots, 'split_seed': r['split_seed'], 'train_seed': r['train_seed'],
                          'probe': probe_stage, 'metric_error': e, 'saved_readout_exact': True}
                if not probe_stage:
                    gate_error = 0.
                    if r['method'] in BASELINES:
                        fresh = replay_baseline_embedding(r['method'], state, model, x, edges)
                        fresh_rebuilt = fresh
                    elif r['method'].startswith('F100'):
                        detail['prefit_order_guard'] = verify_order_diagnostic(folder, state, r['audit'])
                        p = state['prompt']; gate = p['gate'].to(device); tokens = p['tokens'].to(device)
                        fresh = model.conv_layers[1](old['act'](hidden+10.*(gate@tokens.tanh())*hidden.std(0).clamp_min(1e-6)), edges)
                        cache_key = ('F100', r['support_sha256'])
                        if cache_key not in route_cache:
                            raw_scores = old['rectified_teacher'](raw_attrs-raw_attrs.mean(0, keepdim=True), s, y, classes, 'rect_teacher_ref50', r['train_seed'])[0]
                            route_cache[cache_key] = old['temperature_gate'](raw_scores, adj, .1, 'raw')[0]
                        new_gate = route_cache[cache_key]; gate_error = float((new_gate-gate).abs().max())
                        fresh_rebuilt = model.conv_layers[1](old['act'](hidden+10.*(new_gate@tokens.tanh())*hidden.std(0).clamp_min(1e-6)), edges)
                    elif r['method'].startswith('None__'):
                        fresh = fresh_rebuilt = base
                    else:
                        p = state['prompt']; gate = p['route'].to(device); tokens = p['tokens'].to(device)
                        scale = p['scale'].to(device)
                        fresh = model.conv_layers[1](F.leaky_relu(hidden+10.*(gate@tokens.tanh())*scale), edges)
                        kind = r['method'].split('__')[0]; cache_key = (kind, r['support_sha256'])
                        if cache_key not in route_cache:
                            route_cache[cache_key] = routing(kind, x, edges, hidden, s, y, classes)[0]
                        new_gate = route_cache[cache_key]; gate_error = float((new_gate-gate).abs().max())
                        fresh_rebuilt = model.conv_layers[1](F.leaky_relu(hidden+10.*(new_gate@tokens.tanh())*scale), edges)
                    raw_error = float((fresh-z).abs().max())
                    mixed = float(((fresh-z).abs()/(2e-6+1e-6*z.abs())).max())
                    full_scores = saved_readout(state, fresh_rebuilt, s, y, classes, device)
                    full_prob = full_scores[q].cpu().double().softmax(-1).numpy()
                    detail.update(gate_max_error=gate_error, fixed_route_raw_error=raw_error, mixed_forward_ratio=mixed,
                                  predeclared_gate_forward_passed=gate_error < 2e-6 and mixed < 1,
                                  fresh_full_probability_error=float(np.max(abs(full_prob-probabilities))),
                                  fresh_full_class_changes=int(np.count_nonzero(full_prob.argmax(1) != probabilities.argmax(1))),
                                  prompt_gradient_zero=bool(r['audit'].get('prompt_parameters', 0) and r['audit'].get('prompt_gradient_max', 0) == 0))
            details.append(detail)
        assert seen == expected, (dirname, len(seen), len(expected))
    main = [d for d in details if not d['probe']]
    summary = dict(checkpoint_id=job['id'], shots=shots, smoke=args.smoke, main_units=len(main),
        probe_units=len(details)-len(main), maximum_metric_error=metric_error, saved_readout_exact=True,
        predeclared_gate_forward_failures=sum(not d['predeclared_gate_forward_passed'] for d in main),
        fresh_full_class_changes=sum(d['fresh_full_class_changes'] for d in main), details=details,
        original_argmax_assertion_failed_units=sum(not d['prefit_order_guard']['original_argmax_assertion_passed']
            for d in main if 'prefit_order_guard' in d),
        accepted_exact_tie_rows=sum(d['prefit_order_guard']['accepted_exact_tie_rows'] for d in main if 'prefit_order_guard' in d),
        plan_sha256=plan['plan_sha256'])
    target = args.plan.parent/('smoke_analysis' if args.smoke else 'analysis')/(job['id']+'_'+str(shots)+'_audit.json')
    save_json(target, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'details'}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True); p.add_argument('--job-index', type=int, required=True)
    p.add_argument('--shots', type=int, required=True); p.add_argument('--device', default='cuda:0'); p.add_argument('--smoke', action='store_true')
    audit(p.parse_args())
