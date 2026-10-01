"""Complete controls and exact parent references; retain null and negative effects."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from prprompt.protocol import save_json
from experiments.f100_boundary_20261002.prepare import read_plan
from experiments.f100_boundary_20261002.run import parent_methods

METRICS = ['accuracy', 'macro_f1', 'micro_f1', 'roc_auc_macro', 'average_precision_macro']


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def summarize(workspace):
    plan = read_plan(workspace / 'plan.json')
    output = workspace / 'analysis'
    output.mkdir(exist_ok=True)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    counts = {}
    for dirname in ['results', 'references']:
        expected = {(j['id'], k, s, t, m) for j in plan['jobs'] for k in plan['shots']
            for s in plan['split_seeds'] for t in plan['train_seeds']
            for m in (['F100_'+c for c in plan['controls']] if dirname == 'results' else parent_methods(k))}
        seen = set()
        for p in sorted((workspace / dirname).glob('*/*/*/*/*/metrics.json')):
            r = json.loads(p.read_text())
            key = tuple(r[k] for k in ['checkpoint_id', 'shots', 'split_seed', 'train_seed', 'method'])
            assert key in expected and key not in seen and not r['smoke']
            assert r['plan_sha256'] == (plan['plan_sha256'] if dirname == 'results' else plan['parent_plan_sha256'])
            seen.add(key)
            groups[(r['checkpoint_id'], r['shots'], r['method'])][r['split_seed']].append(r['metrics'])
        assert seen == expected
        counts[dirname] = len(seen)
    table, blocks = [], {}
    jobmap = {j['id']: j for j in plan['jobs']}
    for (jid, shots, method), draws in sorted(groups.items()):
        assert len(draws) == 20 and all(len(v) == 3 for v in draws.values())
        b = {s: {m: float(np.mean([r[m] for r in rows])) for m in METRICS} for s, rows in draws.items()}
        blocks[(jid, shots, method)] = b
        row = dict(dataset=jobmap[jid]['dataset'], family='Edgepred_Gprompt', checkpoint_id=jid, shots=shots, method=method,
                   support_draws=20, training_repeats=3, reference_only=method in parent_methods(shots))
        for metric in METRICS:
            values = [r[metric] for r in b.values()]
            row[metric+'_mean'] = float(np.mean(values))
            row[metric+'_std_over_support_draws'] = float(np.std(values, ddof=1))
        table.append(row)
    rng = np.random.default_rng(plan['bootstrap_seed'])
    effects = []
    for j in plan['jobs']:
        for shots in plan['shots']:
            soft, frozen = parent_methods(shots)
            pairs = [(method, ref) for method in ['F100_'+c for c in plan['controls']] for ref in [soft, frozen]] + [(soft, frozen)]
            for ours, ref in pairs:
                a, b = blocks[(j['id'], shots, ours)], blocks[(j['id'], shots, ref)]
                for metric in METRICS:
                    delta = np.array([a[s][metric]-b[s][metric] for s in sorted(a)])
                    ci = np.quantile(rng.choice(delta, (10000, 20), replace=True).mean(1), [.025, .975])
                    effects.append(dict(dataset=j['dataset'], family='Edgepred_Gprompt', checkpoint_id=j['id'], shots=shots,
                        method=ours, reference=ref, metric=metric, mean_delta=float(delta.mean()), ci95_low=float(ci[0]), ci95_high=float(ci[1])))
    audits = [json.loads((output / (j['id']+'_'+str(k)+'_audit.json')).read_text()) for j in plan['jobs'] for k in plan['shots']]
    assert counts == dict(results=3780, references=2520)
    assert sum(a['control_units'] for a in audits) == 3780 and sum(a['reference_units'] for a in audits) == 2520
    scientific = dict(plan_sha256=plan['plan_sha256'], parent_plan_sha256=plan['parent_plan_sha256'],
        controls_present=3780, references_present=2520, missing=[], hashes_pairing_and_saved_readout_passed=True,
        maximum_metric_error=max(a['maximum_metric_error'] for a in audits),
        predeclared_gate_forward_failures=sum(a['predeclared_gate_forward_failures'] for a in audits),
        fresh_full_class_changes=sum(a['fresh_full_class_changes'] for a in audits),
        hardmatched_degenerate=sum(a['hardmatched_degenerate'] for a in audits),
        pre_fit_filter_assert_failure_count=sum(a['pre_fit_filter_assert_failure_count'] for a in audits))
    scientific.update(original_argmax_assertion_failed_units=sum(a['original_argmax_assertion_failed_units'] for a in audits),
        accepted_exact_tie_rows=sum(a['accepted_exact_tie_rows'] for a in audits))
    save_json(output / 'scientific_audit.json', scientific)
    save_json(output / 'method_comparison.json', table)
    save_json(output / 'paired_effects.json', effects)
    write_csv(output / 'method_comparison.csv', table)
    write_csv(output / 'paired_effects.csv', effects)
    lines = ['# F100配方：软分配路由边界', '',
        '3780个新控制单元；2520份原Soft/冻结读出封存参考为复制，未重新拟合。',
        '均值±标准差为%，先平均3个训练重复，再统计20个支持抽样。区间未做多重比较修正。', '',
        'HardMatched退化为零时不能解释为排除了幅度差；1-shot采用明确标注的训练损失适配。', '']
    for j in plan['jobs']:
        for shots in plan['shots']:
            lines += [f"## {j['dataset']} / {shots}-shot", '', '| 方法 | Accuracy | Macro-F1 | Micro-F1 | ROC-AUC | AP |', '|---|---:|---:|---:|---:|---:|']
            for r in table:
                if r['checkpoint_id'] != j['id'] or r['shots'] != shots: continue
                cells = [f"{100*r[m+'_mean']:.3f} ± {100*r[m+'_std_over_support_draws']:.3f}" for m in METRICS]
                lines.append('| '+r['method']+' | '+' | '.join(cells)+' |')
            lines += ['']
    lines += ['## 数值与工程资格', '', '```json', json.dumps(scientific, ensure_ascii=False, indent=2), '```', '']
    (output / '完整结果.md').write_text('\n'.join(lines))
    print(json.dumps(scientific), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('workspace', type=Path)
    summarize(p.parse_args().workspace)
