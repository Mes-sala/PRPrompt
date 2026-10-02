#!/usr/bin/env python3
"""Extract the user-selected PRPrompt recipe and matched controls, without fitting.

All source records are checked against the completed delivery manifests. Original
paired intervals are retained; only the previously absent frozen-linear comparison
gets a new, explicitly descriptive bootstrap using the existing sampling protocol.
Run with the local NumPy runtime. Checkpoints are never read or downloaded.
"""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'Actor', 'Chameleon']
METRICS = ['accuracy', 'macro_f1', 'micro_f1', 'roc_auc_macro', 'average_precision_macro']
MAIN = ['FrozenPrototype', 'FrozenLinear', 'FullFineTune', 'GPF', 'GPFPlus', 'GraphPromptDiag', 'PRPrompt']
ABLATION = ['FrozenPrototype', 'FrozenLinear', 'NoRouting', 'RoutedLinear', 'PRPrompt']
LABELS = {
    'FrozenPrototype': '冻结预训练＋原型分类', 'FrozenLinear': '冻结预训练＋线性分类',
    'FullFineTune': '全量微调', 'GPF': 'GPF', 'GPFPlus': 'GPF-plus',
    'GraphPromptDiag': 'GraphPrompt（统一适配）', 'PRPrompt': 'PRPrompt',
    'NoRouting': '无节点相关门控（共享提示）',
    'RoutedLinear': 'PRPrompt 表示＋单独训练线性分类器',
}
DIRECT = {'None__CenteredPrototype': 'FrozenPrototype', 'None__Linear': 'FrozenLinear',
          'FullFineTune': 'FullFineTune', 'GPF': 'GPF', 'GPFPlus': 'GPFPlus', 'GraphPromptDiag': 'GraphPromptDiag'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def canonical(method):
    if method in ['F100Historical', 'F100OneShotAdapt']:
        return 'PRPrompt'
    return DIRECT.get(method)


def build(output, receipt_dir):
    assert not output.exists(), 'Do not overwrite a delivered paper package.'
    roots = {'main': ROOT/'experiments/outputs/boundary_5090_20261002_R3',
             'aux': ROOT/'experiments/outputs/F100_route_boundary_5090_20261002_R3'}
    delivery_paths = {'main': ROOT/'experiments/local/boundary_20261002_R3/independent_results_only_full_audit.json',
                      'aux': ROOT/'experiments/local/f100_boundary_20261002_R3/independent_results_only_full_audit.json'}
    manifests, plans, evidence = {}, {}, {}
    for key, root in roots.items():
        delivery = read(delivery_paths[key])
        assert delivery['phase'] == 'complete' and delivery['smoke'] is False
        assert delivery['paired_hashes_passed'] and delivery['support_query_and_nested_shots_passed']
        assert delivery['sha256_mismatches'] == 0 and delivery['local_model_binaries_present'] == 0
        assert sha(root/'result_files_sha256.json') == delivery['result_manifest_sha256']
        plans[key] = read(root/'plan.json')
        assert plans[key]['plan_sha256'] == delivery['plan_sha256']
        plan_payload = {k: v for k, v in plans[key].items() if k != 'plan_sha256'}
        assert hashlib.sha256(json.dumps(plan_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == plans[key]['plan_sha256']
        manifests[key] = read(root/'result_files_sha256.json')
        evidence[key] = dict(plan_sha256=delivery['plan_sha256'],
                             result_manifest_sha256=delivery['result_manifest_sha256'],
                             delivery_audit_sha256=sha(delivery_paths[key]))
    assert plans['aux']['parent_plan_sha256'] == plans['main']['plan_sha256']
    implementation_hashes = {}
    for key, names in {
        'main': ['prprompt/train.py', 'prprompt/model.py', 'experiments/boundary_20261002/model.py',
                 'experiments/boundary_20261002/probes.py', 'experiments/v1_f100/historical.py',
                 'experiments/v1_f100/reference/run_round83_graft.py', 'experiments/v1_f100/reference/gate79.py'],
        'aux': ['experiments/f100_boundary_20261002/model.py'],
    }.items():
        for name in names:
            actual = sha(ROOT/name)
            assert actual == plans[key]['source'][name], name
            implementation_hashes[name] = actual
    checked = {}

    def checked_json(key, relative):
        path = roots[key]/relative
        digest = sha(path)
        assert digest == manifests[key][relative], (key, relative)
        checked[key+'/'+relative] = digest
        return read(path)

    main_rows = checked_json('main', 'analysis/results_method_comparison.json')
    probe_rows = checked_json('main', 'analysis/probes_method_comparison.json')
    aux_rows = checked_json('aux', 'analysis/method_comparison.json')
    main_effects = checked_json('main', 'analysis/paired_effects.json')
    aux_effects = checked_json('aux', 'analysis/paired_effects.json')
    rows, records, original = [], {}, {}
    for stage, source_rows, key in [('results', main_rows, 'main'), ('probes', probe_rows, 'main'), ('results', aux_rows, 'aux')]:
        for row in source_rows:
            if key == 'aux':
                name = 'NoRouting' if row['method'] == 'F100_Shared' else None
            elif stage == 'probes':
                name = 'RoutedLinear' if row['method'] == 'F100__Linear' else None
            else:
                name = canonical(row['method'])
            if name is None:
                continue
            target = {k: v for k, v in row.items() if k not in ['method', 'reference_only']}
            target.update(method=name, label=LABELS[name])
            k = (row['dataset'], row['shots'], name)
            assert k not in original
            original[k] = (key, stage, row['method'])
            rows.append(target)
            unit_rows = []
            for split in plans['main']['split_seeds']:
                for seed in plans['main']['train_seeds']:
                    rel = f"{stage}/{row['checkpoint_id']}/{row['shots']}/{split}/{seed}/{row['method']}/metrics.json"
                    unit = checked_json(key, rel)
                    assert not unit['smoke'] and unit['plan_sha256'] == plans[key]['plan_sha256']
                    assert (unit['dataset'], unit['shots'], unit['split_seed'], unit['train_seed'], unit['method']) == (row['dataset'], row['shots'], split, seed, row['method'])
                    if name == 'RoutedLinear':
                        assert unit['fixed_embedding'] and not unit['embedding_updated'] and not unit['query_labels_received']
                    else:
                        assert not unit['audit']['query_labels_received']
                    unit_rows.append(unit)
                    records[(row['dataset'], row['shots'], name, split, seed)] = unit
            assert len(unit_rows) == 60
            for metric in METRICS:
                values = [statistics.mean(u['metrics'][metric] for u in unit_rows if u['split_seed'] == split)
                          for split in plans['main']['split_seeds']]
                assert abs(statistics.mean(values)-row[metric+'_mean']) < 1e-12
                assert abs(statistics.stdev(values)-row[metric+'_std_over_support_draws']) < 1e-12
    assert len(rows) == 189 and len(records) == 11340
    row_map = {(r['dataset'], r['shots'], r['method']): r for r in rows}
    # Align all methods to the exact parent units, not merely equal seeds.
    pair_fields = ['checkpoint_sha256', 'graph_sha256', 'split_policy_sha256', 'support_sha256',
                   'query_sha256', 'support_labels_sha256', 'evaluation_labels_sha256']
    for (dataset, shot, name, split, seed), u in records.items():
        parent = records[(dataset, shot, 'PRPrompt', split, seed)]
        assert all(u[f] == parent[f] for f in pair_fields)
        if name == 'RoutedLinear':
            assert u['parent_state_sha256'] == parent['state_sha256']
            assert u['embedding_sha256'] == parent['embedding_sha256']
    # Confirm that the fixed-representation prototype probe is exactly the final readout.
    for r in probe_rows:
        if r['method'] == 'F100__CenteredPrototype':
            parent = row_map[(r['dataset'], r['shots'], 'PRPrompt')]
            for metric in METRICS:
                for suffix in ['_mean', '_std_over_support_draws']:
                    assert abs(r[metric+suffix]-parent[metric+suffix]) < 1e-12

    effects = []
    for r in main_effects:
        name, ref = canonical(r['method']), canonical(r['reference'])
        if r['stage'] == 'results' and name == 'PRPrompt' and ref in MAIN[:-1]:
            lo, hi = r['paired_bootstrap_95ci']
        elif r['stage'] == 'probes' and r['method'] == 'F100__CenteredPrototype' and r['reference'] == 'F100__Linear':
            name, ref = 'PRPrompt', 'RoutedLinear'
            lo, hi = r['paired_bootstrap_95ci']
        else:
            continue
        effects.append(dict(dataset=r['dataset'], shots=r['shots'], method=name, reference=ref,
                            metric=r['metric'], mean_delta=r['mean_delta'], ci95_low=lo, ci95_high=hi,
                            interval_source='original_paired_interval'))
    for r in aux_effects:
        if canonical(r['method']) == 'PRPrompt' and r['reference'] == 'None__CenteredPrototype':
            ref, delta, lo, hi = 'FrozenPrototype', r['mean_delta'], r['ci95_low'], r['ci95_high']
        elif r['method'] == 'F100_Shared' and canonical(r['reference']) == 'PRPrompt':
            ref, delta, lo, hi = 'NoRouting', -r['mean_delta'], -r['ci95_high'], -r['ci95_low']
        else:
            continue
        effects.append(dict(dataset=r['dataset'], shots=r['shots'], method='PRPrompt', reference=ref,
                            metric=r['metric'], mean_delta=delta, ci95_low=lo, ci95_high=hi,
                            interval_source='original_paired_interval'))
    # This comparison was absent from the original paired table. Reuse the
    # declared 20 support blocks x mean of 3 repeats, 10000 percentile draws.
    rng = np.random.default_rng(20261002)
    for dataset in DATASETS:
        for shot in [1, 3, 5]:
            for metric in METRICS:
                delta = np.array([statistics.mean(records[(dataset, shot, 'PRPrompt', s, t)]['metrics'][metric] -
                                                   records[(dataset, shot, 'FrozenLinear', s, t)]['metrics'][metric]
                                                   for t in [0, 1, 2]) for s in plans['main']['split_seeds']])
                lo, hi = np.quantile(rng.choice(delta, (10000, 20), replace=True).mean(1), [.025, .975])
                effects.append(dict(dataset=dataset, shots=shot, method='PRPrompt', reference='FrozenLinear',
                                    metric=metric, mean_delta=float(delta.mean()), ci95_low=float(lo), ci95_high=float(hi),
                                    interval_source='new_descriptive_paired_bootstrap_20261002'))
    assert len(effects) == 840
    effect_keys = {(r['dataset'], r['shots'], r['reference'], r['metric']) for r in effects}
    assert len(effect_keys) == len(effects)
    for r in effects:
        expected = row_map[(r['dataset'], r['shots'], 'PRPrompt')][r['metric']+'_mean'] - row_map[(r['dataset'], r['shots'], r['reference'])][r['metric']+'_mean']
        assert abs(expected-r['mean_delta']) < 1e-12

    # Scope fresh-forward diagnostics to the selected methods only.
    diagnostics = []
    for key in ['main', 'aux']:
        for job in plans[key]['jobs']:
            for shot in [1, 3, 5]:
                rel = f"analysis/{job['id']}_{shot}_audit.json"
                audit = checked_json(key, rel)
                for detail in audit['details']:
                    if key == 'aux':
                        name = 'NoRouting' if detail['method'] == 'F100_Shared' else None
                    elif detail.get('probe'):
                        name = 'RoutedLinear' if detail['method'] == 'F100__Linear' else None
                    else:
                        name = canonical(detail['method'])
                    if name:
                        assert detail['saved_readout_exact']
                        fields = ['split_seed', 'train_seed', 'saved_readout_exact', 'metric_error',
                                  'gate_max_error', 'fixed_route_raw_error', 'mixed_forward_ratio',
                                  'predeclared_gate_forward_passed', 'fresh_full_probability_error',
                                  'fresh_full_class_changes', 'prefit_order_guard']
                        diagnostics.append(dict(dataset=job['dataset'], shots=shot,
                                                **{k: detail[k] for k in fields if k in detail}, method=name))
    assert len(diagnostics) == len(records)
    diag_summary = []
    for name in MAIN + ['NoRouting', 'RoutedLinear']:
        cases = [d for d in diagnostics if d['method'] == name]
        assert len(cases) == 1260
        diag_summary.append(dict(method=name, units=len(cases), saved_readout_exact=True,
                                 gate_forward_failed_units=sum(d.get('predeclared_gate_forward_passed') is False for d in cases),
                                 fresh_query_class_changes=sum(d.get('fresh_full_class_changes', 0) for d in cases),
                                 fresh_forward_not_applicable=name == 'RoutedLinear'))
    aggregate = []
    for name in MAIN + ['NoRouting', 'RoutedLinear']:
        selected = [r for r in rows if r['method'] == name]
        aggregate.append(dict(method=name, label=LABELS[name], conditions=21,
                              **{m: 100*statistics.mean(r[m+'_mean'] for r in selected) for m in METRICS},
                              by_shot={str(s): {m: 100*statistics.mean(r[m+'_mean'] for r in selected if r['shots'] == s)
                                                 for m in METRICS} for s in [1, 3, 5]}))
    paired_summary = []
    for ref in MAIN[:-1] + ['NoRouting', 'RoutedLinear']:
        for metric in METRICS:
            chosen = [r for r in effects if r['reference'] == ref and r['metric'] == metric]
            assert len(chosen) == 21
            paired_summary.append(dict(reference=ref, metric=metric, conditions=21,
                                       mean_delta_pp=100*statistics.mean(r['mean_delta'] for r in chosen),
                                       mean_positive=sum(r['mean_delta'] > 0 for r in chosen),
                                       mean_negative=sum(r['mean_delta'] < 0 for r in chosen),
                                       ci_positive=sum(r['ci95_low'] > 0 for r in chosen),
                                       ci_negative=sum(r['ci95_high'] < 0 for r in chosen),
                                       ci_contains_zero=sum(r['ci95_low'] <= 0 <= r['ci95_high'] for r in chosen)))

    output.mkdir(parents=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    save(output/'主比较数据.json', [r for r in rows if r['method'] in MAIN])
    save(output/'消融数据.json', [r for r in rows if r['method'] in ABLATION])
    save(output/'配对差值与区间.json', effects)
    save(output/'汇总统计.json', dict(methods=aggregate, paired=paired_summary))
    save(output/'数值复核.json', dict(methods=diag_summary, cases=[d for d in diagnostics if
        d.get('predeclared_gate_forward_passed') is False or d.get('fresh_full_class_changes', 0) > 0],
        thresholds_changed=False, sealed_predictions_changed=False,
        qualification='保存表示上的读出精确通过；新 CUDA 全链前向并非逐位相同。类别变化按实验单元累计，不是独立节点数。'))
    # The final method's individual results are explicitly extracted and renamed.
    with (output/'PRPrompt逐次结果.jsonl').open('w') as f:
        for k, u in sorted(records.items()):
            if k[2] != 'PRPrompt':
                continue
            r = {x: u[x] for x in ['dataset', 'shots', 'split_seed', 'train_seed', 'checkpoint_id',
                                   'checkpoint_sha256', 'graph_sha256', 'support_sha256', 'query_sha256',
                                   'support_labels_sha256', 'predictions_sha256', 'selection_sha256', 'metrics']}
            r.update(method='PRPrompt', one_shot_loss_adapted=u['shots'] == 1)
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    save(receipt_dir/'checked_source_files.json', checked)
    save(output/'数据来源与验收.json', dict(schema='prprompt-final-paper-extraction-v1',
        selected_recipe='F100Soft', naming='PRPrompt', user_selected_after_completed_experiments=True,
        source=evidence, source_method_mapping={
            'PRPrompt': {'1-shot': 'F100OneShotAdapt', '3/5-shot': 'F100Historical'},
            'FrozenPrototype': 'None__CenteredPrototype', 'FrozenLinear': 'None__Linear',
            'NoRouting': 'F100_Shared', 'RoutedLinear': 'F100__Linear (fixed parent embedding)'},
        source_files_checked=len(checked), source_file_receipt_sha256=sha(receipt_dir/'checked_source_files.json'),
        unique_selected_units=11340, final_method_units=1260, main_table_rows=147, ablation_table_rows=105,
        main_units=8820, additional_shared_prompt_units=1260, conditional_linear_probe_units=1260,
        pairing_hashes_verified=True, aggregate_values_recomputed=True, existing_intervals_preserved=735,
        new_frozen_linear_intervals=105, bootstrap=dict(seed=20261002, draws=10000, support_blocks=20,
        repeats_averaged_first=3, scope='fixed graph/checkpoint descriptive', multiple_comparison_correction=False),
        source_runtime_revision='0328a2bb68f249a34c89557178b6233daf5ce20d',
        implementation_source_sha256=implementation_hashes,
        extraction_script_sha256=sha(Path(__file__)), numpy_version=np.__version__,
        machine_verification='passed', human_verification='not_recorded', submission_ready=False,
        fit_or_hyperparameter_changes=False, per_condition_recipe_selection=False,
        model_binaries_copied=False))

    def mean_cell(dataset, shot, name, metric):
        r = row_map[(dataset, shot, name)]
        return f"{100*r[metric+'_mean']:.2f} ± {100*r[metric+'_std_over_support_draws']:.2f}"

    protocol = '单位为 %；先平均每次支持抽样的 3 个训练种子，再报告 20 次支持抽样的均值 ± 样本标准差。所有方法使用同一图、checkpoint、支持集和查询集。'
    lines = ['# PRPrompt 主比较', '', protocol, '',
             '冻结预训练＋原型分类采用与 PRPrompt 相同的最终表示中心化及原型读出。GPF、GPF-plus、GraphPrompt 为本项目统一节点协议实现；GraphPrompt 为对角读出提示适配，不能解释成全部原论文最佳配置复现。', '',
             '## 跨条件描述性均值', '',
             table(['方法', 'Acc', 'Macro-F1', '1-shot Acc', '3-shot Acc', '5-shot Acc'],
                   [[LABELS[a['method']], f"{a['accuracy']:.2f}", f"{a['macro_f1']:.2f}"] +
                    [f"{a['by_shot'][str(s)]['accuracy']:.2f}" for s in [1, 3, 5]] for a in aggregate if a['method'] in MAIN]), '',
             '以下为全部 21 个条件；跨条件均值不代表独立图上的总体推断。']
    for metric, title in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')]:
        lines += ['', '## '+title, '', table(['数据集 / shot'] + [LABELS[m] for m in MAIN],
                  [[f'{d} / {s}']+[mean_cell(d, s, m, metric) for m in MAIN] for d in DATASETS for s in [1, 3, 5]])]
    lines += ['', '## 结果边界', '',
              'Photo、Computers 的 3/5-shot 中 PRPrompt 落后于全量微调，Accuracy 与 Macro-F1 的原配对区间均为负。Actor 三个 shot 相对微调的 Accuracy 区间均包含零。不得用跨条件平均掩盖这些条件。', '',
              '五指标完整数值见[主比较数据](主比较数据.json)，配对区间见[配对比较](配对比较.md)，复跑限制见[数值复核](数值复核.json)。']
    (output/'主比较表.md').write_text('\n'.join(lines)+'\n')

    lines = ['# 原型路由消融与分类器对照', '', protocol, '',
             '## 对照定义', '',
             table(['对照', '改动与检验目标'], [
                 [LABELS['NoRouting'], '固定每个节点的系数为 1/C；保留同一类别提示矩阵、注入位置、尺度、损失、优化器和读出。检验节点相关原型路由相对共享提示的价值。'],
                 [LABELS['FrozenPrototype'], '完全去掉提示，冻结预训练编码器并保持相同原型读出。检验内部表示适配的整体价值。'],
                 [LABELS['FrozenLinear'], '不使用提示，仅在冻结预训练表示上训练线性分类器。检验单独训练分类器是否足够。'],
                 [LABELS['RoutedLinear'], '先按完整方法训练提示，再冻结该提示和最终表示，只拟合线性分类器。检验同一表示的读出差异；不是提示与线性头联合训练。']]), '',
             '共享提示保留名义 token 参数量，但所有节点相同的组合降低了有效自由度，不能宣称完全排除了容量差异。固定表示分类器的选参条件于已经使用全部支持标签训练的父表示，不能称为整个流程的严格嵌套验证。', '',
             '## 跨条件描述性均值', '',
             table(['方法', 'Acc', 'Macro-F1'], [[LABELS[n]]+[f"{next(a for a in aggregate if a['method']==n)[m]:.2f}" for m in ['accuracy', 'macro_f1']] for n in ABLATION]), '',
             '## PRPrompt 相对对照的配对收益', '',
             table(['对照', '指标', '平均差值 / pp', '均值正 / 负', '区间正 / 负 / 含零'],
                   [[LABELS[r['reference']], r['metric'], f"{r['mean_delta_pp']:+.2f}", f"{r['mean_positive']} / {r['mean_negative']}",
                     f"{r['ci_positive']} / {r['ci_negative']} / {r['ci_contains_zero']}"]
                    for r in paired_summary if r['reference'] in ABLATION[:-1] and r['metric'] in METRICS[:2]])]
    for metric, title in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')]:
        lines += ['', '## '+title+'：全部条件', '', table(['数据集 / shot']+[LABELS[n] for n in ABLATION],
                  [[f'{d} / {s}']+[mean_cell(d, s, n, metric) for n in ABLATION] for d in DATASETS for s in [1, 3, 5]])]
    lines += ['', '这些对照支持完整原型路由提示相对共享提示和冻结读出的价值，不单独证明每个路由构造步骤各自有效，也不证明优于所有可学习路由。']
    (output/'消融与分类器对照.md').write_text('\n'.join(lines)+'\n')
    effect_map = {(r['dataset'], r['shots'], r['reference'], r['metric']): r for r in effects}
    lines = ['# PRPrompt 配对比较', '',
             '差值为 PRPrompt − 对照，单位为百分点。方括号为按支持抽样配对的描述性 95% bootstrap 区间，未作多重比较校正。包含零不等于等效。', '',
             '除冻结预训练＋线性分类器的 105 个五指标区间为本次按原统计协议补算外，其余 735 个区间保留原记录；反向比较仅翻转差值和区间端点。']
    for ref in MAIN[:-1]+['NoRouting', 'RoutedLinear']:
        lines += ['', '## '+LABELS[ref], '']
        cells = []
        for d in DATASETS:
            for s in [1, 3, 5]:
                vals = []
                for m in METRICS[:2]:
                    e = effect_map[(d, s, ref, m)]
                    vals.append(f"{100*e['mean_delta']:+.2f} [{100*e['ci95_low']:+.2f}, {100*e['ci95_high']:+.2f}]")
                cells.append([f'{d} / {s}']+vals)
        lines += [table(['数据集 / shot', 'Accuracy', 'Macro-F1'], cells)]
    (output/'配对比较.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output), selected_units=len(records), aggregate_rows=len(rows),
                          paired_effects=len(effects), diagnostics=diag_summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'docs/paper/PRPrompt_20261002')
    p.add_argument('--receipt-dir', type=Path, default=ROOT/'experiments/local/paper_PRPrompt_20261002')
    args = p.parse_args()
    build(args.output, args.receipt_dir)
