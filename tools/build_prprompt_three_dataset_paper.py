#!/usr/bin/env python3
"""Build the requested three-dataset paper scope from sealed results.

No training, classifier probes, or new bootstrap intervals are introduced.
The complete earlier material remains an immutable source, not the current draft.
"""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['Cora', 'CiteSeer', 'PubMed']
METRICS = ['accuracy', 'macro_f1', 'micro_f1', 'roc_auc_macro', 'average_precision_macro']
MAIN = ['FrozenPrototype', 'FullFineTune', 'GPF', 'GPFPlus', 'GraphPromptDiag', 'PRPrompt']
ABLATION = ['FrozenPrototype', 'NoRouting', 'PRPrompt']
LABELS = {'FrozenPrototype': '冻结预训练', 'FullFineTune': '全量微调', 'GPF': 'GPF',
          'GPFPlus': 'GPF-plus', 'GraphPromptDiag': 'GraphPrompt（统一适配）',
          'PRPrompt': 'PRPrompt', 'NoRouting': '无节点相关门控（共享提示）'}
METHOD_MAP = {'FrozenPrototype': 'None__CenteredPrototype', 'FullFineTune': 'FullFineTune',
              'GPF': 'GPF', 'GPFPlus': 'GPFPlus', 'GraphPromptDiag': 'GraphPromptDiag',
              'NoRouting': 'F100_Shared'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, r))+' |' for r in rows])


def build(source, output, receipt_dir):
    assert not output.exists(), 'Use a new output directory; do not overwrite a delivered package.'
    source_manifest = read(source/'文件校验清单.json')
    for name, digest in source_manifest.items():
        assert sha(source/name) == digest, name
    parent = read(source/'数据来源与验收.json')
    roots = {'main': ROOT/'experiments/outputs/boundary_5090_20261002_R3',
             'aux': ROOT/'experiments/outputs/F100_route_boundary_5090_20261002_R3'}
    manifests = {}
    for key, root in roots.items():
        assert sha(root/'result_files_sha256.json') == parent['source'][key]['result_manifest_sha256']
        manifests[key] = read(root/'result_files_sha256.json')
    checked = {}

    def checked_json(key, relative):
        path = roots[key]/relative
        digest = sha(path)
        assert digest == manifests[key][relative], (key, relative)
        checked[key+'/'+relative] = digest
        return read(path)

    main = [dict(r, label=LABELS[r['method']]) for r in read(source/'主比较数据.json')
            if r['dataset'] in DATASETS and r['method'] in MAIN]
    ablation = [dict(r, label=LABELS[r['method']]) for r in read(source/'消融数据.json')
                if r['dataset'] in DATASETS and r['method'] in ABLATION]
    assert len(main) == 54 and len(ablation) == 27
    rows = {}
    for r in main+ablation:
        k = (r['dataset'], r['shots'], r['method'])
        assert k not in rows or rows[k] == r
        rows[k] = r
    assert len(rows) == 63
    units = {}
    for (dataset, shot, name), row in rows.items():
        key = 'aux' if name == 'NoRouting' else 'main'
        method = ('F100OneShotAdapt' if shot == 1 else 'F100Historical') if name == 'PRPrompt' else METHOD_MAP[name]
        for split in range(30093000, 30093020):
            for seed in [0, 1, 2]:
                relative = f"results/{row['checkpoint_id']}/{shot}/{split}/{seed}/{method}/metrics.json"
                u = checked_json(key, relative)
                assert (u['dataset'], u['shots'], u['method'], u['split_seed'], u['train_seed']) == (dataset, shot, method, split, seed)
                assert not u['smoke'] and not u['audit']['query_labels_received']
                assert u['plan_sha256'] == parent['source'][key]['plan_sha256']
                units[(dataset, shot, name, split, seed)] = u
    assert len(units) == 3780
    max_error = 0.
    for (dataset, shot, name), row in rows.items():
        for metric in METRICS:
            vals = [statistics.mean(units[(dataset, shot, name, s, t)]['metrics'][metric] for t in [0, 1, 2])
                    for s in range(30093000, 30093020)]
            errs = [abs(statistics.mean(vals)-row[metric+'_mean']),
                    abs(statistics.stdev(vals)-row[metric+'_std_over_support_draws'])]
            assert max(errs) < 1e-12
            max_error = max(max_error, *errs)
    for (d, s, n, split, seed), u in units.items():
        ref = units[(d, s, 'PRPrompt', split, seed)]
        for field in ['checkpoint_sha256', 'graph_sha256', 'support_sha256', 'query_sha256',
                      'support_labels_sha256', 'split_policy_sha256', 'evaluation_labels_sha256']:
            assert u[field] == ref[field]
    effects = [r for r in read(source/'配对差值与区间.json')
               if r['dataset'] in DATASETS and r['reference'] in MAIN[:-1]+['NoRouting']]
    assert len(effects) == 270
    assert len({(r['dataset'], r['shots'], r['reference'], r['metric']) for r in effects}) == 270
    for r in effects:
        assert r['interval_source'] == 'original_paired_interval'
        delta = rows[r['dataset'], r['shots'], 'PRPrompt'][r['metric']+'_mean'] - rows[r['dataset'], r['shots'], r['reference']][r['metric']+'_mean']
        assert abs(delta-r['mean_delta']) < 1e-12
    raw = [json.loads(line) for line in (source/'PRPrompt逐次结果.jsonl').read_text().splitlines()
           if json.loads(line)['dataset'] in DATASETS]
    assert len(raw) == 540
    for r in raw:
        u = units[r['dataset'], r['shots'], 'PRPrompt', r['split_seed'], r['train_seed']]
        for field in ['metrics', 'predictions_sha256', 'support_sha256', 'query_sha256', 'checkpoint_sha256']:
            assert r[field] == u[field]
    diagnostics = []
    for d in DATASETS:
        for shot in [1, 3, 5]:
            jid = rows[d, shot, 'PRPrompt']['checkpoint_id']
            for key in ['main', 'aux']:
                audit = checked_json(key, f'analysis/{jid}_{shot}_audit.json')
                for detail in audit['details']:
                    if detail.get('probe'):
                        continue
                    method = detail['method']
                    if key == 'aux':
                        name = 'NoRouting' if method == 'F100_Shared' else None
                    else:
                        name = ('PRPrompt' if method in ['F100Historical', 'F100OneShotAdapt'] else
                                next((n for n, m in METHOD_MAP.items() if m == method and n in MAIN), None))
                    if name is None:
                        continue
                    assert detail['saved_readout_exact']
                    diagnostics.append(dict(dataset=d, shots=shot, method=name, **{f: detail[f] for f in
                        ['split_seed', 'train_seed', 'saved_readout_exact', 'gate_max_error', 'mixed_forward_ratio',
                         'predeclared_gate_forward_passed', 'fresh_full_probability_error', 'fresh_full_class_changes']}))
    assert len(diagnostics) == len(units)
    diag_summary = []
    for n in MAIN+['NoRouting']:
        cases = [r for r in diagnostics if r['method'] == n]
        assert len(cases) == 540
        diag_summary.append(dict(method=n, units=540, saved_readout_exact=True,
                                 gate_forward_failed_units=sum(not r['predeclared_gate_forward_passed'] for r in cases),
                                 fresh_query_class_changes=sum(r['fresh_full_class_changes'] for r in cases)))
    output.mkdir(parents=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    save(output/'主比较数据.json', main)
    save(output/'路由消融数据.json', ablation)
    save(output/'配对差值与区间.json', effects)
    save(output/'条件内统计.json', dict(aggregation='Each dataset, shot and method is kept separate; average training seeds within each support draw, then report mean and sample standard deviation across support draws.', conditions=list(rows.values())))
    save(output/'数值复核.json', dict(methods=diag_summary,
         cases=[r for r in diagnostics if not r['predeclared_gate_forward_passed'] or r['fresh_full_class_changes']],
         thresholds_changed=False, sealed_predictions_changed=False,
         qualification='本范围全部保存读出精确通过，原门控／前向阈值全部通过，未观察到新前向类别变化；不据此声称所有浮点值逐位一致。'))
    (output/'PRPrompt逐次结果.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in raw))
    save(receipt_dir/'checked_source_files.json', checked)
    save(output/'数据来源与验收.json', dict(schema='prprompt-three-dataset-paper-v1',
        datasets=DATASETS, shots=[1, 3, 5], selected_recipe='F100Soft', display_method='PRPrompt',
        reporting_scope='User-selected after completed experiments; descriptive subset, not an independent confirmatory study.',
        source_material_directory=str(source.relative_to(ROOT)), source_material_manifest_sha256=sha(source/'文件校验清单.json'),
        source=parent['source'], runtime_source_revision=parent['source_runtime_revision'],
        implementation_source_sha256={k: v for k, v in parent['implementation_source_sha256'].items() if not k.endswith('/probes.py')},
        selected_main_methods=MAIN, selected_ablation_methods=ABLATION, main_units=3240, shared_prompt_units=540,
        unique_units=3780, final_method_units=540, main_rows=54, ablation_rows=27, unique_aggregate_rows=63,
        paired_effects=270, source_files_checked=len(checked), source_file_receipt_sha256=sha(receipt_dir/'checked_source_files.json'),
        maximum_reaggregation_error=max_error, pairing_hashes_verified=True, existing_intervals_unchanged=True,
        new_bootstrap=False, new_training=False, model_binaries_copied=False,
        extraction_script_sha256=sha(Path(__file__)), human_verification='not_recorded', submission_ready=False))

    def cell(d, s, n, m):
        r = rows[d, s, n]
        return f"{100*r[m+'_mean']:.2f} ± {100*r[m+'_std_over_support_draws']:.2f}"

    protocol = '仅 Cora、CiteSeer、PubMed，1/3/5-shot。单位为 %；先平均 3 个训练种子，再统计 20 次支持抽样的均值 ± 样本标准差。'
    lines = ['# PRPrompt 三数据集主比较', '', protocol, '',
             '冻结预训练采用与 PRPrompt 相同的预测预处理和原型读出。GPF、GPF-plus、GraphPrompt 为本项目统一节点协议实现，不能描述为各原论文最佳配置复现。', '',
             '所有均值仅在同一数据集、同一 shot、同一方法内计算，不跨数据集或实验设置汇总。']
    for metric, title in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')]:
        lines += ['', '## '+title, '', table(['数据集 / shot']+[LABELS[n] for n in MAIN],
                  [[f'{d} / {s}']+[cell(d, s, n, metric) for n in MAIN] for d in DATASETS for s in [1, 3, 5]])]
    lines += ['', '本表只支持所列三个数据集和固定预训练 checkpoint 下的结论。九个条件不等于九张独立图；本次范围整理不构成新的独立确认实验。', '',
              '完整五指标见[主比较数据](主比较数据.json)，配对区间见[配对比较](配对比较.md)。']
    (output/'主比较表.md').write_text('\n'.join(lines)+'\n')
    lines = ['# 原型路由消融', '', protocol, '',
             '## 对照定义', '', table(['方法', '定义'], [
                 ['PRPrompt', '完整原型路由提示，冻结编码器，采用原型分类。'],
                 [LABELS['NoRouting'], '所有节点路由设为 1/C，保留同一提示参数矩阵、注入位置、尺度、训练损失、优化器及原型读出。'],
                 ['无提示（冻结预训练）', '完全去掉提示，保持冻结编码器与相同最终原型读出。']]), '',
             '共享提示保留名义参数量，但均匀组合降低了有效自由度；该对照检验整套路由的作用，不能单独排除容量差异。', '',
             '所有数值按数据集与 shot 分别报告；配对差值及区间见[配对比较](配对比较.md)。']
    for metric, title in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')]:
        lines += ['', '## '+title, '', table(['数据集 / shot']+[LABELS[n] for n in ABLATION],
                  [[f'{d} / {s}']+[cell(d, s, n, metric) for n in ABLATION] for d in DATASETS for s in [1, 3, 5]])]
    lines += ['', '结论限于原型路由提示整体相对共享提示及无提示的价值；没有分别隔离校正、中心化和图聚合步骤的贡献。']
    (output/'路由消融.md').write_text('\n'.join(lines)+'\n')
    index = {(r['dataset'], r['shots'], r['reference'], r['metric']): r for r in effects}
    lines = ['# 三数据集配对比较', '', '差值为 PRPrompt − 对照，单位为百分点。全部区间保留原配对支持抽样 95% bootstrap 记录，未重新计算或进行多重比较校正。']
    for ref in MAIN[:-1]+['NoRouting']:
        values = []
        for d in DATASETS:
            for s in [1, 3, 5]:
                cells = []
                for m in METRICS[:2]:
                    r = index[d, s, ref, m]
                    cells.append(f"{100*r['mean_delta']:+.2f} [{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}]")
                values.append([f'{d} / {s}']+cells)
        lines += ['', '## '+LABELS[ref], '', table(['数据集 / shot', 'Accuracy', 'Macro-F1'], values)]
    (output/'配对比较.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output), selected_units=len(units), own_units=len(raw),
                          condition_rows=len(rows), numerical=diag_summary), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT/'docs/paper/PRPrompt_20261002')
    p.add_argument('--output', type=Path, default=ROOT/'docs/paper/PRPrompt_三数据集_20261002')
    p.add_argument('--receipt-dir', type=Path, default=ROOT/'experiments/local/paper_PRPrompt_three_20261002')
    a = p.parse_args()
    build(a.source, a.output, a.receipt_dir)
