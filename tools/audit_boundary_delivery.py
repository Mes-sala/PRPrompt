"""Independent local NumPy audit. Smoke mode never claims formal completion or delivery.

Full mode requires a fresh remote exit observation and hashes every manifest file,
then checks every prediction/identity, reaggregates tables and reproduces paired CIs.
No training, deletion, network calls or query-driven selection occurs here.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.metric_utils import numpy_metrics, sha

METRICS = ['accuracy', 'macro_f1', 'micro_f1', 'roc_auc_macro', 'average_precision_macro']
DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'Actor', 'Chameleon']
HEADS = ['Linear', 'Prototype', 'CenteredPrototype', 'Ridge']
ROUTES = ['Soft', 'Hard', 'HardMatched', 'Shared', 'None']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    names = {'int64': 'torch.int64', 'int32': 'torch.int32', 'float32': 'torch.float32',
             'float64': 'torch.float64', 'bool': 'torch.bool', 'uint8': 'torch.uint8'}
    # Match the frozen runner's dtype/shape prefix without importing Torch.
    header = str((names[value.dtype.name], tuple(value.shape))).encode()
    return hashlib.sha256(header+value.tobytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def methods(shots):
    return [r+'__'+h for r in ROUTES for h in HEADS]+['FullFineTune', 'GPF', 'GPFPlus', 'GraphPromptDiag']+[
        'F100OneShotAdapt' if shots == 1 else 'F100Historical']


def main_pairs(shots):
    historical = 'F100OneShotAdapt' if shots == 1 else 'F100Historical'
    pairs = [(f'Soft__{h}', f'{r}__{h}') for h in HEADS for r in ['HardMatched', 'Hard', 'Shared', 'None']]
    pairs += [(f'{r}__{h}', f'{r}__Linear') for r in ROUTES for h in HEADS[1:]]
    pairs += [(historical, ref) for ref in ['Soft__Linear', 'Soft__CenteredPrototype', 'FullFineTune']]
    pairs += [(ours, ref) for ours in ['Soft__Linear', 'Soft__CenteredPrototype', 'Soft__Ridge', historical]
              for ref in ['FullFineTune', 'GPF', 'GPFPlus', 'GraphPromptDiag']]
    return set(pairs)


def audit(a, report):
    root = a.workspace.resolve()
    plan = json.loads((root/'plan.json').read_text())
    assert plan['plan_sha256'] == a.expected_plan
    assert digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) == a.expected_plan
    supplement = plan['schema'] == 'f100-paired-routing-boundary-v1'
    assert [j['dataset'] for j in plan['jobs']] == DATASETS
    assert all(j['family'] == 'Edgepred_Gprompt' for j in plan['jobs'])
    assert plan['shots'] == [1, 3, 5] and plan['split_seeds'] == list(range(30093000, 30093020)) and plan['train_seeds'] == [0, 1, 2]
    if supplement:
        assert plan['controls'] == ['Hard', 'HardMatched', 'Shared'] and plan['expected_units'] == 3780
    else:
        assert all(plan['methods_by_shot'][str(k)] == methods(k) for k in plan['shots'])
        assert plan['readout_probe_sources'] == ['Soft__Linear', 'Soft__CenteredPrototype', 'F100']
        assert plan['probe_heads'] == HEADS and plan['expected_units'] == 31500 and plan['expected_probe_units'] == 15120
    report.update(plan_sha256=a.expected_plan, smoke=a.smoke, supplementary=supplement, formal_completion_claimed=False)
    hashes = {}
    if not a.smoke:
        status = json.loads((root/'pipeline_status.json').read_text())
        observation = json.loads(a.server_observation.read_text())
        assert status['phase'] == 'server_complete' and observation['phase'] == 'server_complete'
        assert observation['plan_sha256'] == a.expected_plan and observation['pid'] == status['pid']
        assert observation['actual_controller_alive'] is False and observation['actual_worker_count'] == 0
        assert sha(root/'result_files_sha256.json') == observation['result_manifest_sha256']
        manifest = json.loads((root/'result_files_sha256.json').read_text())
        hashes.update(manifest)
        hashes.update(observation['extra_files_sha256'])
        for index, (name, expected) in enumerate(hashes.items(), 1):
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert sha(root/relative) == expected, name
            if index % 10000 == 0: print(json.dumps(dict(stage='file_hashes', checked=index)), flush=True)
        report.update(file_hashes_checked=len(hashes), sha256_mismatches=0, remote_exit_observation_sha256=sha(a.server_observation))
    for name, expected in plan['source'].items(): assert sha(root/'source'/name) == expected, name
    labels = {}
    for name, spec in plan['datasets'].items():
        assert sha(root/'dataset_snapshots'/Path(spec['snapshot']).name) == spec['snapshot_sha256']
        with np.load((a.audit_reference or root/'audit_reference')/(name+'.npz')) as data:
            labels[name] = {k: data[k].copy() for k in ['labels', 'pool', 'query']}
            labels[name]['expected_support'] = {k: data[k].copy() for k in data.files if k.startswith('support_')}
        for key in ['labels', 'pool', 'query']: assert array_hash(labels[name][key]) == spec[key+'_sha256']
        assert not np.intersect1d(labels[name]['pool'], labels[name]['query']).size
    for job in plan['jobs']: assert sha(root/'checkpoints'/Path(job['checkpoint']).name) == job['checkpoint_sha256']
    splits = plan['split_seeds'][:1] if a.smoke else plan['split_seeds']
    seeds = plan['train_seeds'][:1] if a.smoke else plan['train_seeds']
    prefixes = ['smoke', 'smoke_references'] if supplement and a.smoke else ['results', 'references'] if supplement else ['smoke', 'smoke_probes'] if a.smoke else ['results', 'probes']
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    pairing, nested, checked, max_error = {}, {}, {}, 0.
    jobmap = {j['id']: j for j in plan['jobs']}
    parent_manifest = None
    if supplement and not a.smoke:
        assert a.parent_workspace and a.parent_delivery
        parent_delivery = json.loads(a.parent_delivery.read_text())
        assert parent_delivery['phase'] == 'complete' and not parent_delivery['smoke']
        assert parent_delivery['plan_sha256'] == plan['parent_plan_sha256']
        parent_manifest = json.loads((a.parent_workspace/'result_files_sha256.json').read_text())
        assert parent_delivery['result_manifest_sha256'] == sha(a.parent_workspace/'result_files_sha256.json')
    for prefix in prefixes:
        references = prefix.endswith('references')
        probe = prefix.endswith('probes')
        names = lambda k: (['F100_'+c for c in plan['controls']] if supplement and not references else
            ['F100OneShotAdapt' if k == 1 else 'F100Historical', 'None__CenteredPrototype'] if references else
            [p+'__'+h for p in plan['readout_probe_sources'] for h in HEADS] if probe else methods(k))
        expected = {(j['id'], k, s, t, m) for j in plan['jobs'] for k in plan['shots'] for s in splits for t in seeds for m in names(k)}
        seen = set()
        for p in sorted((root/prefix).glob('*/*/*/*/*/metrics.json')):
            r = json.loads(p.read_text())
            unit = tuple(r[k] for k in ['checkpoint_id', 'shots', 'split_seed', 'train_seed', 'method'])
            assert unit in expected and unit not in seen
            seen.add(unit)
            job = jobmap[r['checkpoint_id']]
            assert r['dataset'] == job['dataset'] and r['checkpoint_sha256'] == job['checkpoint_sha256']
            spec = plan['datasets'][job['dataset']]
            assert r['graph_sha256'] == spec['graph_sha256']
            assert r['split_policy_sha256'] == digest(dict(pool=spec['pool_sha256'], query=spec['query_sha256']))
            assert r['family'] == 'Edgepred_Gprompt' and r['smoke'] == a.smoke
            assert r['plan_sha256'] == (plan['parent_plan_sha256'] if references else a.expected_plan)
            f = p.parent
            seal = json.loads((f/'seal.json').read_text())
            payload = {k: v for k, v in seal.items() if k not in ['signature', 'predictions_sha256', 'support_sha256', 'query_sha256']}
            assert digest(payload) == r['signature'] == seal['signature']
            for key in ['plan_sha256', 'predictions_sha256', 'support_sha256', 'query_sha256']: assert seal[key] == r[key]
            for name, key in [('predictions.npz', 'predictions_sha256'), ('selection.json', 'selection_sha256')]+([] if references else [('state.pt.gz', 'state_sha256')]):
                relative = str((f/name).relative_to(root))
                assert (hashes[relative] if relative in hashes else sha(f/name)) == r[key]
            assert not json.loads((f/'selection.json').read_text())['query_labels_used']
            with np.load(f/'predictions.npz') as data:
                prob, s, q = (data[k].copy() for k in ['probabilities', 'support', 'query'])
            l = labels[job['dataset']]
            assert np.array_equal(q, l['query']) and np.isin(s, l['pool']).all() and not np.intersect1d(s, q).size
            assert len(np.unique(s)) == len(s) and (np.bincount(l['labels'][s], minlength=prob.shape[1]) == r['shots']).all()
            if l['expected_support']:
                assert np.array_equal(s, l['expected_support'][f"support_{r['shots']}_{r['split_seed']}"])
            assert prob.shape == (len(q), plan['datasets'][job['dataset']]['classes']) and np.isfinite(prob).all()
            assert (prob >= 0).all() and np.max(np.abs(prob.sum(1)-1)) < 1e-12
            assert array_hash(s) == r['support_sha256'] and array_hash(q) == r['query_sha256']
            assert array_hash(l['labels'][s]) == r['support_labels_sha256'] and array_hash(l['labels'][q]) == r['evaluation_labels_sha256']
            metrics = numpy_metrics(l['labels'][q], prob)
            error = max(abs(metrics[k]-r['metrics'][k]) for k in METRICS)
            assert error < 1e-12, str(p)
            max_error = max(max_error, error)
            identity = tuple(r[k] for k in ['checkpoint_sha256', 'graph_sha256', 'split_policy_sha256', 'support_sha256', 'query_sha256', 'support_labels_sha256'])
            pairkey = (r['checkpoint_id'], r['shots'], r['split_seed'])
            assert pairkey not in pairing or pairing[pairkey] == identity
            pairing[pairkey] = identity
            nkey = (r['checkpoint_id'], r['split_seed'], r['shots'])
            assert nkey not in nested or nested[nkey] == set(s.tolist())
            nested[nkey] = set(s.tolist())
            if probe:
                assert r['fixed_embedding'] and not r['embedding_updated'] and not r['query_labels_received']
                parent = root/('smoke' if a.smoke else 'results')/r['checkpoint_id']/str(r['shots'])/str(r['split_seed'])/str(r['train_seed'])/r['parent_method']
                pr = json.loads((parent/'metrics.json').read_text())
                assert sha(parent/'metrics.json') == r['parent_metrics_sha256']
                assert pr['state_sha256'] == r['parent_state_sha256'] and pr['embedding_sha256'] == r['embedding_sha256']
            else:
                assert not r['audit']['query_labels_received']
            if references and parent_manifest is not None:
                relative = Path('results')/p.relative_to(root/'references')
                for name in ['metrics.json', 'seal.json', 'selection.json', 'predictions.npz']:
                    expected_sha = parent_manifest[str(relative.parent/name)]
                    assert sha(f/name) == expected_sha
            stage = 'references' if references else 'probes' if probe else 'results'
            groups[(stage, r['checkpoint_id'], r['shots'], r['method'])][r['split_seed']].append(metrics)
            if len(seen) % 500 == 0: print(json.dumps(dict(stage=prefix, metrics_checked=len(seen))), flush=True)
        assert seen == expected, (prefix, len(seen), len(expected), list(expected-seen)[:5])
        checked[prefix] = len(seen)
    for job in plan['jobs']:
        for split in splits: assert nested[(job['id'], split, 1)] < nested[(job['id'], split, 3)] < nested[(job['id'], split, 5)]
    report.update(unit_counts=checked, maximum_independent_metric_error=max_error, paired_hashes_passed=True,
                  support_query_and_nested_shots_passed=True, result_manifest_sha256=sha(root/'result_files_sha256.json') if not a.smoke else None)
    server_audits = []
    for job in plan['jobs']:
        for shots in plan['shots']:
            path = root/('smoke_analysis' if a.smoke else 'analysis')/(job['id']+'_'+str(shots)+'_audit.json')
            sr = json.loads(path.read_text())
            assert sr['plan_sha256'] == a.expected_plan and sr['checkpoint_id'] == job['id'] and sr['shots'] == shots and sr['smoke'] == a.smoke
            if supplement:
                required = {(s, t, 'F100_'+c) for s in splits for t in seeds for c in plan['controls']}
                assert sr['control_units'] == len(required) and sr['reference_units'] == len(splits)*len(seeds)*2
                actual = {(d['split_seed'], d['train_seed'], d['method']) for d in sr['details']}
                main_details = sr['details']
            else:
                required = {(s, t, m, False) for s in splits for t in seeds for m in methods(shots)}
                required |= {(s, t, parent+'__'+h, True) for s in splits for t in seeds for parent in plan['readout_probe_sources'] for h in HEADS}
                actual = {(d['split_seed'], d['train_seed'], d['method'], d['probe']) for d in sr['details']}
                main_details = [d for d in sr['details'] if not d['probe']]
                assert sr['main_units'] == len(main_details) and sr['probe_units'] == len(required)-len(main_details)
            assert actual == required and len(sr['details']) == len(required)
            assert all(d['saved_readout_exact'] for d in sr['details'])
            assert sr['predeclared_gate_forward_failures'] == sum(not d['predeclared_gate_forward_passed'] for d in main_details)
            assert sr['fresh_full_class_changes'] == sum(d['fresh_full_class_changes'] for d in main_details)
            server_audits.append(sr)
    report.update(server_every_unit_readout_audit_coverage_verified=True,
                  predeclared_gate_forward_failures=sum(r['predeclared_gate_forward_failures'] for r in server_audits),
                  fresh_full_class_changes=sum(r['fresh_full_class_changes'] for r in server_audits),
                  original_numerical_acceptance_passed=all(r['predeclared_gate_forward_failures'] == 0 for r in server_audits))
    if not a.smoke:
        blocks = {}
        for key, draws in groups.items():
            assert len(draws) == 20 and all(len(v) == 3 for v in draws.values())
            blocks[key] = {s: {m: float(np.mean([r[m] for r in rows])) for m in METRICS} for s, rows in draws.items()}
        maximum_table_error = 0.
        tables = [('analysis/method_comparison.csv', None)] if supplement else [('analysis/results_method_comparison.csv', 'results'), ('analysis/probes_method_comparison.csv', 'probes')]
        table_seen = set()
        for filename, stage in tables:
            for r in csv.DictReader((root/filename).open()):
                kind = ('references' if r['reference_only'] == 'True' else 'results') if supplement else stage
                key = (kind, r['checkpoint_id'], int(r['shots']), r['method'])
                assert key in blocks and key not in table_seen
                table_seen.add(key)
                for metric in METRICS:
                    v = [b[metric] for b in blocks[key].values()]
                    maximum_table_error = max(maximum_table_error, abs(float(np.mean(v))-float(r[metric+'_mean'])),
                        abs(float(np.std(v, ddof=1))-float(r[metric+'_std_over_support_draws'])))
        assert table_seen == set(blocks) and maximum_table_error < 1e-12
        effects = json.loads((root/'analysis/paired_effects.json').read_text())
        expected_effects = set()
        for j in plan['jobs']:
            for k in plan['shots']:
                soft = 'F100OneShotAdapt' if k == 1 else 'F100Historical'
                sets = {'results': ({(m, ref) for m in ['F100_'+c for c in plan['controls']] for ref in [soft, 'None__CenteredPrototype']} | {(soft, 'None__CenteredPrototype')}) if supplement else main_pairs(k)}
                if not supplement: sets['probes'] = {(p+'__'+h, p+'__Linear') for p in plan['readout_probe_sources'] for h in HEADS[1:]}
                for stage, pairs in sets.items():
                    for ours, ref in pairs:
                        for metric in METRICS: expected_effects.add((stage, j['id'], k, ours, ref, metric))
        seen_effects = set()
        rng = np.random.default_rng(plan['bootstrap_seed'])
        maximum_effect_error = 0.
        for r in effects:
            stage = 'results' if supplement else r['stage']
            key = (stage, r['checkpoint_id'], r['shots'], r['method'], r['reference'], r['metric'])
            assert key in expected_effects and key not in seen_effects
            seen_effects.add(key)
            ours_stage = 'references' if supplement and r['method'] in ['F100Historical', 'F100OneShotAdapt'] else stage
            ref_stage = 'references' if supplement else stage
            x = blocks[(ours_stage, r['checkpoint_id'], r['shots'], r['method'])]
            y = blocks[(ref_stage, r['checkpoint_id'], r['shots'], r['reference'])]
            delta = np.array([x[s][r['metric']]-y[s][r['metric']] for s in sorted(x)])
            ci = np.quantile(rng.choice(delta, (10000, 20), replace=True).mean(1), [.025, .975])
            saved_ci = [r['ci95_low'], r['ci95_high']] if supplement else r['paired_bootstrap_95ci']
            maximum_effect_error = max(maximum_effect_error, abs(float(delta.mean())-r['mean_delta']), float(np.max(abs(ci-saved_ci))))
        assert seen_effects == expected_effects and maximum_effect_error < 1e-12
        report.update(method_rows_reaggregated=len(table_seen), paired_metric_effects_checked=len(seen_effects),
                      maximum_table_error=maximum_table_error, maximum_paired_effect_or_interval_error=maximum_effect_error,
                      formal_completion_claimed=True)
    report.update(phase='smoke_audit_complete' if a.smoke else 'complete', finished=time.time())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('workspace', type=Path)
    p.add_argument('--expected-plan', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--server-observation', type=Path)
    p.add_argument('--parent-workspace', type=Path)
    p.add_argument('--parent-delivery', type=Path)
    p.add_argument('--audit-reference', type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    record = dict(started=time.time(), workspace=str(args.workspace.resolve()), phase='checking', smoke=args.smoke)
    try:
        assert args.smoke or args.server_observation, 'Full audit requires a fresh verified remote terminal observation'
        audit(args, record)
    except Exception as error:
        record.update(phase='failed', error=repr(error), finished=time.time())
        save(args.output, record)
        raise
    save(args.output, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
