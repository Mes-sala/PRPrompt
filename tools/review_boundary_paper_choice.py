"""Summarize complete audited recipes without new tests or per-condition selection.

Cross-condition means and CI sign counts are descriptive. The original support
cluster bootstrap intervals are reused; no pooled interval or p-value is made.
"""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


PRIMARY = ('accuracy', 'macro_f1')
DATASETS = ('Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'Actor', 'Chameleon')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(method):
    return 'F100' if method in ('F100Historical', 'F100OneShotAdapt') else method


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def summarize(values):
    return dict(conditions=len(values), mean_delta_pp=100 * statistics.mean(x[0] for x in values),
                mean_positive=sum(x[0] > 0 for x in values), mean_negative=sum(x[0] < 0 for x in values),
                mean_zero=sum(x[0] == 0 for x in values),
                CI_positive=sum(x[1] > 0 for x in values), CI_negative=sum(x[2] < 0 for x in values),
                CI_crosses_or_touches_zero=sum(x[1] <= 0 <= x[2] for x in values))


def review(main, supplement, report):
    provenance = json.loads((report / 'artifact_provenance.json').read_text())
    assert provenance['main_formal_units'] == 31500
    assert provenance['detached_readout_probes'] == 15120
    assert provenance['F100_new_controls'] == 3780 and provenance['F100_copied_references'] == 2520
    assert sha(main / 'result_files_sha256.json') == provenance['main_result_manifest_sha256']
    assert sha(supplement / 'result_files_sha256.json') == provenance['F100_result_manifest_sha256']
    inputs = {}

    def load(root, prefix, name):
        path = root / 'analysis' / name
        key = prefix + '/' + name
        inputs[key] = sha(path)
        assert inputs[key] == provenance['analysis_json_input_sha256'][key]
        return json.loads(path.read_text())

    rows = load(main, 'main', 'results_method_comparison.json')
    probes = load(main, 'main', 'probes_method_comparison.json')
    effects = load(main, 'main', 'paired_effects.json')
    aux_rows = load(supplement, 'F100', 'method_comparison.json')
    aux_effects = load(supplement, 'F100', 'paired_effects.json')
    assert tuple(map(len, [rows, probes, effects, aux_rows, aux_effects])) == (525, 252, 6090, 105, 735)
    conditions = {(dataset, shot) for dataset in DATASETS for shot in (1, 3, 5)}
    candidates = collections.defaultdict(list)
    candidate_lookup = {}
    for row in rows + [r for r in aux_rows if not r['reference_only']]:
        assert row['family'] == 'Edgepred_Gprompt'
        assert row['support_draws'] == 20 and row['training_repeats'] == 3
        name = label(row['method'])
        key = row['dataset'], row['shots'], name
        assert key not in candidate_lookup
        candidate_lookup[key] = row
        candidates[name].append(row)
    assert len(candidates) == 28
    candidate_means = []
    for name, values in candidates.items():
        assert {(r['dataset'], r['shots']) for r in values} == conditions and len(values) == 21
        candidate_means.append(dict(method=name, conditions=21,
            **{metric + '_mean_percent': 100 * statistics.mean(r[metric + '_mean'] for r in values)
               for metric in PRIMARY},
            by_shot={str(shot): {metric + '_mean_percent': 100 * statistics.mean(
                r[metric + '_mean'] for r in values if r['shots'] == shot) for metric in PRIMARY}
                for shot in (1, 3, 5)}))
    candidate_means.sort(key=lambda r: (-r['accuracy_mean_percent'], r['method']))
    pairing = collections.defaultdict(list)
    by_shot = collections.defaultdict(list)
    for row in effects:
        if row['metric'] not in PRIMARY:
            continue
        method, reference = label(row['method']), label(row['reference'])
        triple = (row['mean_delta'], *row['paired_bootstrap_95ci'])
        key = row['stage'], method, reference, row['metric']
        pairing[key].append(triple)
        by_shot[key + (row['shots'],)].append(triple)
    for row in aux_effects:
        if row['metric'] not in PRIMARY:
            continue
        method, reference = label(row['method']), label(row['reference'])
        delta, lo, hi = row['mean_delta'], row['ci95_low'], row['ci95_high']
        if reference == 'F100':
            method, reference = reference, method
            delta, lo, hi = -delta, -hi, -lo
        key = 'F100_same_recipe', method, reference, row['metric']
        pairing[key].append((delta, lo, hi))
        by_shot[key + (row['shots'],)].append((delta, lo, hi))
    paired_summary = []
    for key, values in sorted(pairing.items()):
        stage, method, reference, metric = key
        assert len(values) == 21
        if stage != 'probes':
            deltas = [candidate_lookup[ds, shot, method][metric + '_mean'] -
                      candidate_lookup[ds, shot, reference][metric + '_mean']
                      for ds, shot in conditions]
            assert abs(statistics.mean(deltas) - statistics.mean(x[0] for x in values)) < 1e-12
        paired_summary.append(dict(stage=stage, method=method, reference=reference, metric=metric,
            **summarize(values), by_shot={str(shot): summarize(by_shot[key + (shot,)]) for shot in (1, 3, 5)}))
    output = dict(schema='complete-boundary-paper-review-descriptive-v1',
        review_source_sha256=sha(Path(__file__)), source_json_sha256=inputs,
        candidate_means=candidate_means, paired_summary=paired_summary,
        independent_graph_samples_claimed=False, new_bootstrap_or_p_values=False,
        per_condition_recipe_stitching=False,
        training_or_hyperparameters_changed=False,
        qualification='21 conditions are seven fixed checkpoints crossed with three nested shot levels. '
            'Unweighted means and CI sign counts describe this grid, not independent-graph inference. '
            'Original support-cluster intervals are uncorrected for multiplicity. F100 at one shot is '
            'the predeclared inclusive-loss adaptation, not the historical leave-one-out loss.')
    target = report / '方案统计汇总.json'
    doc = report / '方案统计汇总.md'
    assert not target.exists() and not doc.exists(), 'Preserve existing review outputs.'
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    text = ['# 完整方案的描述性统计', '',
        '每个方案固定应用于七个数据集的1/3/5-shot，共21个条件。跨条件等权均值仅用于描述，'
        '不把嵌套shot或同一固定图视作独立图样本，不生成新的跨图区间或p值。', '',
        'F100的1-shot为预声明的inclusive适配，3/5-shot为原LOO历史配方。'
        '数值资格和微调负结果见结果总结；本表不能替代这些限制。', '',
        '## 固定方案的全条件均值', '']
    text += table(['方案', 'Accuracy均值(%)', 'Macro-F1均值(%)',
                   '1-shot Acc', '3-shot Acc', '5-shot Acc'], [
        [r['method'], f"{r['accuracy_mean_percent']:.2f}", f"{r['macro_f1_mean_percent']:.2f}"] +
        [f"{r['by_shot'][str(s)]['accuracy_mean_percent']:.2f}" for s in (1, 3, 5)]
        for r in candidate_means])
    text += ['', '## 原配对区间的方向计数', '',
        '差值为方案减对照，单位百分点。正/负区间指原95%区间完全位于零的正/负侧；'
        '区间含零不代表等效。区间未作多重比较校正，不对这些方向计数重新做显著性检验。', '']
    for stage in ('results', 'probes', 'F100_same_recipe'):
        text += ['### ' + stage, '']
        text += table(['方案 − 对照', '指标', '平均差值(pp)', '均值正/负/零', '区间正/负/含零'], [
            [r['method'] + ' − ' + r['reference'], r['metric'], f"{r['mean_delta_pp']:+.2f}",
             f"{r['mean_positive']}/{r['mean_negative']}/{r['mean_zero']}",
             f"{r['CI_positive']}/{r['CI_negative']}/{r['CI_crosses_or_touches_zero']}"]
            for r in paired_summary if r['stage'] == stage])
        text += ['']
    text += ['探针Linear的支持内CV条件于已使用全部支持标签拟合的父表示，'
             '属于条件读出对照，不能替代完整流程的嵌套验证。', '']
    doc.write_text('\n'.join(text))
    print(json.dumps(dict(candidate_recipes=28, paired_primary_summaries=len(paired_summary),
                          new_inferential_statistics=False, per_condition_recipe_stitching=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--supplement', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    review(a.main.resolve(), a.supplement.resolve(), a.report.resolve())
