#!/usr/bin/env python3
"""Render completed, independently delivered boundary results without fitting or selection.

Only the audited, serialized tables/effects are used. No new p-values, bootstrap
samples, per-condition method choices, or experiments are generated here.
"""
import argparse
import collections
import hashlib
import json
import shutil
import statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.local_artifact_policy import POLICY, remote_only


METRICS = ['accuracy', 'macro_f1', 'micro_f1', 'roc_auc_macro', 'average_precision_macro']
METRIC_NAMES = dict(zip(METRICS, ['Accuracy', 'Macro-F1', 'Micro-F1', 'macro ROC-AUC', 'macro AP']))
ROUTES = ['Soft', 'Hard', 'HardMatched', 'Shared', 'None']
HEADS = ['Linear', 'Prototype', 'CenteredPrototype', 'Ridge']
PRIMARY = ['accuracy', 'macro_f1']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require_delivery(root, delivery_path, supplementary):
    delivery, plan = read(delivery_path), read(root / 'plan.json')
    assert delivery['phase'] == 'complete' and not delivery['smoke'], 'Full independent delivery is required; smoke is insufficient.'
    assert delivery['workspace'] == str(root.resolve()), 'Delivery belongs to a different workspace.'
    assert bool(delivery['supplementary']) == supplementary
    assert delivery['plan_sha256'] == plan['plan_sha256']
    assert digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) == plan['plan_sha256']
    expected = dict(results=3780, references=2520) if supplementary else dict(results=31500, probes=15120)
    assert delivery['unit_counts'] == expected
    for key in ['paired_hashes_passed', 'support_query_and_nested_shots_passed', 'server_every_unit_readout_audit_coverage_verified']:
        assert delivery[key] is True, key
    assert delivery['local_artifact_policy'] == POLICY
    assert delivery['remote_model_bytes_sha256_verified'] is True
    assert delivery['local_model_binaries_present'] == 0
    assert delivery['remote_only_files_verified'] > 0
    assert delivery['retained_local_files_verified'] > 0
    assert len(delivery['remote_only_receipt_sha256']) == 64
    assert not any(p.is_file() for p in root.rglob('*') if remote_only(p.relative_to(root)))
    if plan.get('prefit_order_guard'):
        assert delivery['prefit_order_policy_units_checked'] == (5040 if supplementary else 1260)
        assert delivery['genuine_order_reversal_rows'] == 0
        assert type(delivery['original_argmax_assertion_failed_units']) is int
        assert type(delivery['accepted_exact_tie_rows']) is int
    manifest_path = root / 'result_files_sha256.json'
    assert sha(manifest_path) == delivery['result_manifest_sha256']
    return plan, read(manifest_path), delivery


def audited_json(root, manifest, relative):
    path = root / relative
    assert relative in manifest and sha(path) == manifest[relative], relative
    return read(path)


def cell_mean(row, metric):
    return f"{100 * row[metric + '_mean']:.3f} ± {100 * row[metric + '_std_over_support_draws']:.3f}"


def cell_effect(row, reverse=False):
    delta = row['mean_delta']
    if 'paired_bootstrap_95ci' in row:
        low, high = row['paired_bootstrap_95ci']
    else:
        low, high = row['ci95_low'], row['ci95_high']
    if reverse:
        delta, low, high = -delta, -high, -low
    return f'{100 * delta:+.3f} [{100 * low:+.3f}, {100 * high:+.3f}]'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def save_text(output, name, lines):
    (output / name).write_text('\n'.join(lines) + '\n')


def resource_summary(root, manifest, expected, expected_plan):
    groups = collections.defaultdict(list)
    units = 0
    for path in sorted((root / 'results').glob('*/*/*/*/*/metrics.json')):
        relative = path.relative_to(root).as_posix()
        assert relative in manifest and sha(path) == manifest[relative]
        row = read(path)
        assert row['plan_sha256'] == expected_plan and not row['smoke']
        groups[(row['dataset'], row['checkpoint_id'], row['shots'], row['method'])].append(row)
        units += 1
    assert units == expected
    result = []
    for (dataset, jid, shots, method), values in sorted(groups.items()):
        assert len(values) == 60
        record = dict(dataset=dataset, checkpoint_id=jid, shots=shots, method=method, units=60)
        for key in ['trainable_parameters', 'prompt_parameters', 'head_parameters', 'backbone_parameters']:
            counts = {r['audit'].get(key) for r in values}
            assert len(counts) == 1, (dataset, shots, method, key)
            record[key] = counts.pop()
        record['mean_total_seconds_per_unit'] = statistics.mean(r['total_seconds'] for r in values)
        record['mean_final_fit_seconds_per_unit'] = statistics.mean(r['audit']['fit_seconds'] for r in values) if all('fit_seconds' in r['audit'] for r in values) else None
        record['unrecorded_parameter_fields'] = [key for key in ['trainable_parameters', 'prompt_parameters', 'head_parameters', 'backbone_parameters'] if record[key] is None]
        record['maximum_peak_allocated_cuda_mib'] = max(r['peak_cuda_bytes'] for r in values) / 1024**2
        record['maximum_peak_reserved_cuda_mib'] = max(r['peak_reserved_cuda_bytes'] for r in values) / 1024**2
        result.append(record)
    return result


def build(main, supplement, main_delivery, supplement_delivery, output):
    plan, manifest, delivery = require_delivery(main, main_delivery, False)
    aux_plan, aux_manifest, aux_delivery = require_delivery(supplement, supplement_delivery, True)
    assert aux_plan['parent_plan_sha256'] == plan['plan_sha256']
    assert plan['shots'] == [1, 3, 5] and len(plan['jobs']) == 7
    assert all(j['family'] == 'Edgepred_Gprompt' for j in plan['jobs'])
    rows = audited_json(main, manifest, 'analysis/results_method_comparison.json')
    probes = audited_json(main, manifest, 'analysis/probes_method_comparison.json')
    effects = audited_json(main, manifest, 'analysis/paired_effects.json')
    aux_rows = audited_json(supplement, aux_manifest, 'analysis/method_comparison.json')
    aux_effects = audited_json(supplement, aux_manifest, 'analysis/paired_effects.json')
    scientific = audited_json(main, manifest, 'analysis/scientific_audit.json')
    aux_scientific = audited_json(supplement, aux_manifest, 'analysis/scientific_audit.json')
    assert (len(rows), len(probes), len(effects), len(aux_rows), len(aux_effects)) == (525, 252, 6090, 105, 735)
    by_row = {(r['checkpoint_id'], r['shots'], r['method']): r for r in rows}
    by_effect = {(r['stage'], r['checkpoint_id'], r['shots'], r['method'], r['reference'], r['metric']): r for r in effects}
    by_aux = {(r['checkpoint_id'], r['shots'], r['method'], r['reference'], r['metric']): r for r in aux_effects}
    assert len(by_row) == len(rows) and len(by_effect) == len(effects) and len(by_aux) == len(aux_effects)
    # Validate every required display before creating the destination.
    jobs = plan['jobs']
    for job in jobs:
        jid = job['id']
        for shot in plan['shots']:
            assert len([r for r in rows if r['checkpoint_id'] == jid and r['shots'] == shot]) == 25
            for metric in PRIMARY:
                for head in HEADS:
                    for route in ROUTES[1:]:
                        assert ('results', jid, shot, 'Soft__' + head, route + '__' + head, metric) in by_effect
                for route in ROUTES:
                    for head in HEADS[1:]:
                        assert ('results', jid, shot, route + '__' + head, route + '__Linear', metric) in by_effect
                for parent in plan['readout_probe_sources']:
                    for head in HEADS[1:]:
                        assert ('probes', jid, shot, parent + '__' + head, parent + '__Linear', metric) in by_effect
                soft = 'F100OneShotAdapt' if shot == 1 else 'F100Historical'
                for control in ['Hard', 'HardMatched', 'Shared']:
                    assert (jid, shot, 'F100_' + control, soft, metric) in by_aux
    resources = dict(main=resource_summary(main, manifest, 31500, plan['plan_sha256']),
                     F100_controls=resource_summary(supplement, aux_manifest, 3780, aux_plan['plan_sha256']),
                     qualification='Per-unit elapsed times include contention from concurrent workers. Total seconds include support selection, I/O and scoring; final-fit seconds exclude CV. Sums are not wall-clock runtime. CUDA peaks include the worker backbone, data, cached filters and allocator reservations; they are not isolated per-method memory measurements. Copied references are not new fits. Unrecorded fields are null; parameter counts and fit-only times are not inferred.')
    assert not output.exists(), 'Choose a new output directory; existing reports are preserved.'
    output.mkdir(parents=True)
    copied = {}
    for root, hashes, prefix, names in [
        (main, manifest, '主实验', ['results_method_comparison.csv', 'probes_method_comparison.csv', 'paired_effects.csv', '完整结果.md']),
        (supplement, aux_manifest, 'F100同配方', ['method_comparison.csv', 'paired_effects.csv', '完整结果.md']),
    ]:
        for name in names:
            relative = 'analysis/' + name
            assert relative in hashes and sha(root / relative) == hashes[relative]
            target = output / (prefix + '_' + name)
            shutil.copyfile(root / relative, target)
            assert sha(target) == hashes[relative]
            copied[target.name] = hashes[relative]

    routing = ['# 软原型分配的收益边界', '',
        '以下差值统一为 Soft − 对照，单位为百分点，方括号为已保存的配对支持抽样95%区间。',
        '区间未做多重比较校正，不把区间包含零解释成等效。五指标原始配对表见主实验_paired_effects.csv。', '',
        'HardMatched只匹配全图路由的整体Frobenius范数，不保证逐节点扰动幅度相同；恒零Hard不能匹配非零Soft。', '']
    heads = ['# 分类头的收益边界', '',
        '差值为相应头 − Linear，单位为百分点；相同路由下的联合训练与相同父表示上的读出分别呈现。',
        'Ridge读取冻结或提示GCN表示，不是AttributeRidge。包含自身的原型训练与历史F100的LOO训练不同。', '',
        '探针的父表示已经用全部支持标签训练；探针Linear的支持CV条件于该表示，不是完整训练流程的嵌套验证。', '']
    shots = ['# 1/3/5-shot 完整表现', '',
        '均值±标准差为百分数：先平均3个训练重复，再对20个支持抽样计算样本标准差。',
        '支持集使用相同随机排列的嵌套前缀。这里展示描述性shot变化，不新增跨shot显著性检验。',
        'F100列在1-shot是F100OneShotAdapt，在3/5-shot是F100Historical；不能将三列视作完全相同的训练损失。', '']
    for metric in PRIMARY:
        routing += ['## ' + METRIC_NAMES[metric], '']
        heads += ['## ' + METRIC_NAMES[metric] + '：联合训练', '']
        for head in HEADS:
            routing += ['### 分类头 ' + head, '']
            display = []
            for j in jobs:
                for shot in plan['shots']:
                    display.append([f"{j['dataset']} / {shot}"] + [cell_effect(by_effect[('results', j['id'], shot, 'Soft__' + head, route + '__' + head, metric)]) for route in ROUTES[1:]])
            routing += table(['数据集 / shot', 'Soft−Hard', 'Soft−HardMatched', 'Soft−Shared', 'Soft−None'], display) + ['']
        routing += ['### 历史 F100 同配方：仅替换路由', '', '此处把原控制−Soft差值反向显示为Soft−控制，区间端点也反向。', '']
        display = []
        for j in jobs:
            for shot in plan['shots']:
                soft = 'F100OneShotAdapt' if shot == 1 else 'F100Historical'
                display.append([f"{j['dataset']} / {shot}"] + [cell_effect(by_aux[(j['id'], shot, 'F100_' + c, soft, metric)], reverse=True) for c in ['Hard', 'HardMatched', 'Shared']])
        routing += table(['数据集 / shot', 'Soft−Hard', 'Soft−HardMatched', 'Soft−Shared'], display) + ['']
        for route in ROUTES:
            heads += ['### 路由 ' + route, '']
            display = [[f"{j['dataset']} / {shot}"] + [cell_effect(by_effect[('results', j['id'], shot, route + '__' + h, route + '__Linear', metric)]) for h in HEADS[1:]] for j in jobs for shot in plan['shots']]
            heads += table(['数据集 / shot', 'Prototype−Linear', 'CenteredPrototype−Linear', 'Ridge−Linear'], display) + ['']
        heads += ['## ' + METRIC_NAMES[metric] + '：固定表示读出', '']
        for parent in plan['readout_probe_sources']:
            heads += ['### 父表示 ' + parent, '']
            display = [[f"{j['dataset']} / {shot}"] + [cell_effect(by_effect[('probes', j['id'], shot, parent + '__' + h, parent + '__Linear', metric)]) for h in HEADS[1:]] for j in jobs for shot in plan['shots']]
            heads += table(['数据集 / shot', 'Prototype−Linear', 'CenteredPrototype−Linear', 'Ridge−Linear'], display) + ['']
        shots += ['## ' + METRIC_NAMES[metric], '']
        for j in jobs:
            shots += ['### ' + j['dataset'], '']
            methods = [r + '__' + h for r in ROUTES for h in HEADS] + ['FullFineTune', 'GPF', 'GPFPlus', 'GraphPromptDiag', 'F100']
            display = []
            for method in methods:
                cells = []
                for shot in plan['shots']:
                    actual = ('F100OneShotAdapt' if shot == 1 else 'F100Historical') if method == 'F100' else method
                    cells.append(cell_mean(by_row[(j['id'], shot, actual)], metric))
                display.append([method] + cells)
            shots += table(['方法', '1-shot', '3-shot', '5-shot'], display) + ['']
    save_text(output, '软分配边界.md', routing)
    save_text(output, '分类头边界.md', heads)
    save_text(output, 'shot完整表现.md', shots)
    audit = dict(main=scientific, F100_same_recipe=aux_scientific,
                 local_delivery_numerical_qualification=dict(main=delivery['original_numerical_acceptance_passed'], F100_same_recipe=aux_delivery['original_numerical_acceptance_passed']))
    if plan.get('prefit_order_guard'):
        fields = ['prefit_order_policy_units_checked', 'original_argmax_assertion_failed_units',
                  'accepted_exact_tie_rows', 'genuine_order_reversal_rows']
        audit['independent_exact_tie_audit'] = dict(main={k:delivery[k] for k in fields},
            supplement_controls_and_copied_parent_references={k:aux_delivery[k] for k in fields},
            qualification='Original argmax failures remain recorded; exact maximum-set intersection uses no tolerance. Supplement audit counts include copied parent references, which are not new fits.')
    (output / '数值与覆盖审计.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    (output / '参数与实测成本.json').write_text(json.dumps(resources, ensure_ascii=False, indent=2) + '\n')
    provenance = dict(report_builder_sha256=sha(Path(__file__)), main_plan_sha256=plan['plan_sha256'],
        F100_plan_sha256=aux_plan['plan_sha256'], main_result_manifest_sha256=delivery['result_manifest_sha256'],
        F100_result_manifest_sha256=aux_delivery['result_manifest_sha256'],
        main_delivery_audit_sha256=sha(main_delivery), F100_delivery_audit_sha256=sha(supplement_delivery),
        analysis_json_input_sha256={
            'main/' + name: manifest['analysis/' + name] for name in ['results_method_comparison.json', 'probes_method_comparison.json', 'paired_effects.json', 'scientific_audit.json']
        } | {'F100/' + name: aux_manifest['analysis/' + name] for name in ['method_comparison.json', 'paired_effects.json', 'scientific_audit.json']},
        source_tables_copied_without_changes=copied, aggregate_rows=dict(main=525, probes=252, F100_controls_and_references=105),
        paired_effect_rows=dict(main_and_probes=6090, F100_same_recipe=735),
        main_formal_units=31500, detached_readout_probes=15120, F100_new_controls=3780, F100_copied_references=2520,
        final_paper_recommendation='pending_review_of_complete_evidence', human_verification_record='not_provided', submission_ready=False,
        new_statistics=False, query_based_method_selection=False)
    provenance['local_delivery_scope'] = {
        'main': {k: delivery.get(k) for k in ['local_artifact_policy', 'remote_only_files_verified',
            'retained_local_files_verified', 'remote_only_receipt_sha256']},
        'F100_controls': {k: aux_delivery.get(k) for k in ['local_artifact_policy', 'remote_only_files_verified',
            'retained_local_files_verified', 'remote_only_receipt_sha256']},
        'qualification': 'Model states, checkpoints and dataset snapshots remain on the server; retained local results are individually hashed. Scientific coverage and paired audits are unchanged.'}
    (output / 'artifact_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    save_text(output, '结果索引.md', ['# 原 Edgepred_Gprompt 完整边界实验', '',
        '只在两个工作区均完整回传并通过独立审计后生成。31,500个主实验单元、15,120个固定表示探针、3,780个F100控制；2,520个父结果引用不重复计作训练。', '',
        '本地回传范围为代码、预测与指标、选参记录、表格、文档和验收证据；checkpoint、训练状态和数据快照留在服务器，验收记录保存其远端逐文件SHA256核对凭证。', '',
        '- [软分配边界](软分配边界.md)', '- [分类头边界](分类头边界.md)', '- [1/3/5-shot表现](shot完整表现.md)',
        '- [主实验全部五指标](主实验_完整结果.md)', '- [F100同配方全部五指标](F100同配方_完整结果.md)',
        '- [数值与覆盖审计](数值与覆盖审计.json)', '- [参数与实测成本](参数与实测成本.json)', '- [来源清单](artifact_provenance.json)', '',
        '固定图、固定checkpoint下的支持抽样描述性比较；预训练种子未知。全图无标签特征与结构可见，拟合只使用支持标签。',
        '1-shot使用预声明超参数，3/5-shot支持内选参。不同协议下的旧5-shot实验不能直接并入本轮统计。',
        '此索引不自动选论文方案；完整的收益、负结果、微调差距、数值资格和机制边界仍需结合证据判断。', '',
        '## 汇总工作流来源', '',
        '使用本地scientific-writing技能约束证据与数值一致性；AI整理不能替代作者核验。',
        'Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). '
        '[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). '
        '作者与当前版本元数据于2026-10-02从arXiv核对。'])
    report_hashes = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'report_files_sha256.json').write_text(json.dumps(report_hashes, ensure_ascii=False, indent=2) + '\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--main-delivery', type=Path, required=True)
    parser.add_argument('--supplement-delivery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.main.resolve(), args.supplement.resolve(), args.main_delivery,
                   args.supplement_delivery, args.output)
    print(json.dumps(result, ensure_ascii=False))
