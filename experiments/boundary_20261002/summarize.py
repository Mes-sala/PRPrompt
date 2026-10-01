"""Only complete datasets enter final tables; retain all paired effects and qualifications."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from prprompt.protocol import save_json
from experiments.boundary_20261002.prepare import read_plan
from experiments.boundary_20261002.model import ROUTES, HEADS

METRICS = ['accuracy', 'macro_f1', 'micro_f1', 'roc_auc_macro', 'average_precision_macro']

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def summarize(workspace):
    plan = read_plan(workspace/'plan.json'); output = workspace/'analysis'; output.mkdir(exist_ok=True)
    tables, all_effects = {}, []; rng = np.random.default_rng(plan['bootstrap_seed'])
    counts = {}; nested = {}
    for dirname in ['results', 'probes']:
        expected = {(j['id'], k, s, t, m) for j in plan['jobs'] for k in plan['shots'] for s in plan['split_seeds'] for t in plan['train_seeds']
                    for m in (plan['methods_by_shot'][str(k)] if dirname == 'results' else [p+'__'+h for p in plan['readout_probe_sources'] for h in HEADS])}
        seen = set(); groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for path in sorted((workspace/dirname).glob('*/*/*/*/*/metrics.json')):
            r = json.loads(path.read_text()); key = tuple(r[k] for k in ['checkpoint_id', 'shots', 'split_seed', 'train_seed', 'method'])
            assert key in expected and key not in seen; seen.add(key)
            assert r['plan_sha256'] == plan['plan_sha256'] and not r['smoke']
            groups[(r['checkpoint_id'], r['shots'], r['method'])][r['split_seed']].append(r['metrics'])
            if dirname == 'results' and r['method'] == 'Soft__Linear' and r['train_seed'] == 0:
                from numpy import load
                with load(path.parent/'predictions.npz') as p:
                    nested[(r['checkpoint_id'], r['split_seed'], r['shots'])] = set(p['support'].tolist())
        assert seen == expected and len(seen) == (plan['expected_units'] if dirname == 'results' else plan['expected_probe_units'])
        counts[dirname] = len(seen); table = []; blocks = {}; jobmap = {j['id']: j for j in plan['jobs']}
        for (jobid, shots, method), draws in sorted(groups.items()):
            assert len(draws) == 20 and all(len(v) == 3 for v in draws.values())
            b = {s: {m: float(np.mean([r[m] for r in reps])) for m in METRICS} for s, reps in draws.items()}
            blocks[(jobid, shots, method)] = b; j = jobmap[jobid]
            row = dict(dataset=j['dataset'], family=j['family'], checkpoint_id=jobid, shots=shots, method=method, support_draws=20, training_repeats=3)
            for m in METRICS:
                values = [r[m] for r in b.values()]
                row[m+'_mean'] = float(np.mean(values)); row[m+'_std_over_support_draws'] = float(np.std(values, ddof=1))
            table.append(row)
        tables[dirname] = table
        write_csv(output/(dirname+'_method_comparison.csv'), table); save_json(output/(dirname+'_method_comparison.json'), table)
        comparisons = []
        if dirname == 'results':
            for h in HEADS:
                for r in ['HardMatched', 'Hard', 'Shared', 'None']:
                    comparisons.append(('routing', 'Soft__'+h, r+'__'+h))
            for r in ROUTES:
                for h in HEADS[1:]: comparisons.append(('head_joint_training', r+'__'+h, r+'__Linear'))
            for ref in ['Soft__Linear', 'Soft__CenteredPrototype', 'FullFineTune']:
                comparisons.append(('historical_recipe', 'F100', ref))
            for ours in ['Soft__Linear', 'Soft__CenteredPrototype', 'Soft__Ridge', 'F100']:
                for ref in ['FullFineTune', 'GPF', 'GPFPlus', 'GraphPromptDiag']:
                    comparisons.append(('method_comparison', ours, ref))
        else:
            for parent in plan['readout_probe_sources']:
                for h in HEADS[1:]: comparisons.append(('fixed_embedding_head', parent+'__'+h, parent+'__Linear'))
        for j in plan['jobs']:
            for shots in plan['shots']:
                used = set()
                for scope, ours, ref in comparisons:
                    if ours == 'F100': ours = 'F100OneShotAdapt' if shots == 1 else 'F100Historical'
                    if (ours, ref) in used: continue
                    used.add((ours, ref)); a, b = blocks[(j['id'], shots, ours)], blocks[(j['id'], shots, ref)]
                    for metric in METRICS:
                        delta = np.array([a[s][metric]-b[s][metric] for s in sorted(a)])
                        ci = np.quantile(rng.choice(delta, (10000, len(delta)), replace=True).mean(1), [.025, .975]).tolist()
                        all_effects.append(dict(stage=dirname, comparison_type=scope, dataset=j['dataset'], family=j['family'], checkpoint_id=j['id'],
                            shots=shots, method=ours, reference=ref, metric=metric, mean_delta=float(delta.mean()), paired_bootstrap_95ci=ci))
    for j in plan['jobs']:
        for s in plan['split_seeds']:
            assert nested[(j['id'], s, 1)] < nested[(j['id'], s, 3)] < nested[(j['id'], s, 5)]
    save_json(output/'paired_effects.json', all_effects)
    write_csv(output/'paired_effects.csv', [{**{k: v for k, v in r.items() if k != 'paired_bootstrap_95ci'},
        'ci95_low': r['paired_bootstrap_95ci'][0], 'ci95_high': r['paired_bootstrap_95ci'][1]} for r in all_effects])
    audits = [json.loads((output/(j['id']+'_'+str(k)+'_audit.json')).read_text()) for j in plan['jobs'] for k in plan['shots']]
    assert sum(a['main_units'] for a in audits) == 31500 and sum(a['probe_units'] for a in audits) == 15120
    scientific = dict(plan_sha256=plan['plan_sha256'], main_expected=31500, main_present=counts['results'], probe_expected=15120,
        probe_present=counts['probes'], nested_shots_passed=True, saved_readout_exact=True,
        maximum_metric_error=max(a['maximum_metric_error'] for a in audits),
        predeclared_gate_forward_failures=sum(a['predeclared_gate_forward_failures'] for a in audits),
        fresh_full_class_changes=sum(a['fresh_full_class_changes'] for a in audits),
        predeclared_numerical_acceptance_passed=all(a['predeclared_gate_forward_failures'] == 0 for a in audits),
        final_paper_choice='Requires reviewing complete paired tables, negative results and numerical qualifications; no query-wise oracle selection.')
    scientific.update(original_argmax_assertion_failed_units=sum(a['original_argmax_assertion_failed_units'] for a in audits),
        accepted_exact_tie_rows=sum(a['accepted_exact_tie_rows'] for a in audits))
    save_json(output/'scientific_audit.json', scientific); save_json(output/'coverage.json', dict(main=31500, probes=15120, missing=[], smoke=False))
    lines = ['# 原 Edgepred_Gprompt：路由、分类头与shot完整结果', '',
        '主实验31,500单元，固定表示分类头探针15,120单元。均值±标准差为%，先平均训练重复，再统计20个支持抽样。', '',
        '全部方法和配对结果保留，区间是固定图/checkpoint的描述性支持抽样区间，未作多重比较校正。', '']
    for stage, table in tables.items():
        lines += ['## '+('主实验' if stage == 'results' else '固定表示分类头探针'), '']
        for j in plan['jobs']:
            for shots in plan['shots']:
                lines += [f"### {j['dataset']} / {shots}-shot", '', '| 方法 | Accuracy | Macro-F1 | Micro-F1 | ROC-AUC | AP |', '|---|---:|---:|---:|---:|---:|']
                for r in table:
                    if r['checkpoint_id'] == j['id'] and r['shots'] == shots:
                        cells = [f"{100*r[m+'_mean']:.3f} ± {100*r[m+'_std_over_support_draws']:.3f}" for m in METRICS]
                        lines.append('| '+r['method']+' | '+' | '.join(cells)+' |')
                lines += ['']
    lines += ['## 审计边界', '', json.dumps(scientific, ensure_ascii=False, indent=2), '',
              '保存表示读出通过不能替代新CUDA全链精确重放；数值超限及类别变化记录在逐checkpoint/shot审计中。',
              'F100OneShotAdapt改变了不可定义的1-shot LOO训练损失，不能称为原F100。分类头交叉改变训练目标，固定表示探针用于区分读出与联合优化收益。', '']
    (output/'完整结果.md').write_text('\n'.join(lines))
    print(json.dumps(scientific), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('workspace', type=Path); summarize(p.parse_args().workspace)
