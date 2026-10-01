"""Aggregate all complete matched runs, including every negative result."""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from prprompt.protocol import save_json
from experiments.f100_multidata.run import read_plan
METRICS = ["accuracy", "macro_f1", "micro_f1", "roc_auc_macro", "average_precision_macro"]


def summarize(workspace):
    plan = read_plan(workspace/"plan.json")
    expected = {(j["id"], s, t, m) for j in plan["jobs"] for s in plan["split_seeds"]
                for t in plan["train_seeds"] for m in plan["methods"]}
    seen = set(); groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted((workspace/"results").glob("*/5/*/*/*/metrics.json")):
        row = json.loads(path.read_text())
        identity = tuple(row[k] for k in ["checkpoint_id", "split_seed", "train_seed", "method"])
        assert identity in expected and identity not in seen; seen.add(identity)
        assert row["plan_sha256"] == plan["plan_sha256"] and not row["smoke"]
        groups[(row["checkpoint_id"], row["method"])][row["split_seed"]].append(row["metrics"])
    assert seen == expected and len(seen) == plan["expected_units"]
    audits = [json.loads((workspace/"analysis"/(j["id"]+"_audit.json")).read_text()) for j in plan["jobs"]]
    assert sum(a["present"] for a in audits) == len(expected)
    assert all(a["present"] == a["expected"] and not a["missing"] and not a["smoke"] for a in audits)
    table = []; blocks = {}; jobmap = {j["id"]: j for j in plan["jobs"]}
    for (job_id, method), draws in sorted(groups.items()):
        assert len(draws) == 20 and all(len(reps) == 3 for reps in draws.values())
        values = {s: {m: float(np.mean([r[m] for r in reps])) for m in METRICS} for s, reps in draws.items()}
        blocks[(job_id, method)] = values; j = jobmap[job_id]
        row = dict(dataset=j["dataset"], family=j["family"], checkpoint_id=job_id, method=method, support_draws=20, training_runs=60)
        for m in METRICS:
            v = [r[m] for r in values.values()]
            row[m+"_mean"] = float(np.mean(v)); row[m+"_std_over_support_draws"] = float(np.std(v, ddof=1))
        table.append(row)
    output = workspace/"analysis"; save_json(output/"method_comparison.json", table)
    with (output/"method_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    rng = np.random.default_rng(plan["bootstrap_seed"]); pairs = []
    for job in plan["jobs"]:
        ours = blocks[(job["id"], "F100")]
        for method in plan["methods"]:
            if method == "F100":
                continue
            other = blocks[(job["id"], method)]
            for metric in METRICS:
                delta = np.array([ours[s][metric]-other[s][metric] for s in sorted(ours)])
                ci = np.quantile(rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1), [.025, .975]).tolist()
                pairs.append(dict(checkpoint_id=job["id"], dataset=job["dataset"], family=job["family"],
                    reference=method, metric=metric, mean_delta=float(delta.mean()), paired_bootstrap_95ci=ci))
    save_json(output/"paired_effects.json", pairs)
    diagnostics = [d for a in audits for d in a["details"]]
    f100 = [d for d in diagnostics if d["method"] == "F100"]
    scientific = dict(expected=len(expected), present=len(seen), missing=[], plan_sha256=plan["plan_sha256"],
        smoke=False, checkpoint_method_rows=len(table), paired_metric_rows=len(pairs),
        all_saved_scores_replay_exact=True, all_five_metrics_independently_recomputed=True,
        maximum_metric_error=max(a["maximum_metric_error"] for a in audits),
        maximum_f100_gate_error=max(a["maximum_gate_error"] for a in audits),
        maximum_f100_serialized_readout_error=max(a["maximum_serialized_f100_readout_error"] for a in audits),
        maximum_f100_mixed_forward_error_ratio=max(a["maximum_mixed_forward_ratio"] for a in audits),
        maximum_independent_state_probability_error=max(d["independent_state_probability_error"] for d in diagnostics),
        independent_state_class_changes=sum(d["independent_state_class_changes"] for d in diagnostics),
        f100_maximum_full_chain_probability_error=max(d["rebuilt_gate_probability_error"] for d in f100),
        f100_full_chain_class_changes=sum(d["rebuilt_gate_class_changes"] for d in f100),
        f100_strict_full_chain_passed=all(d["rebuilt_gate_probability_error"] < 2e-6 and d["rebuilt_gate_class_changes"] == 0 for d in f100),
        raw_f100_absolute_2e6_failures=sum(d["raw_forward_max_error"] >= 2e-6 for d in f100),
        rule="Predeclared serialized state plus mixed forward acceptance; full fresh forward diagnostics disclosed separately.")
    save_json(output/"scientific_audit.json", scientific)
    save_json(output/"coverage.json", dict(expected=len(expected), present=len(seen), missing=[], smoke=False))
    lookup = {(r["checkpoint_id"], r["method"]): r for r in table}
    lines = ["# 未修正 Edgepred＋v1 F100：五个新增数据集", "",
        "本轮5-shot、100步、20个支持抽样、每个抽样3个训练重复，共5400正式单元；GPU smoke另存，不计成绩。每个数据集/预训练来源只有一份原始权重，预训练种子未知。", "",
        "均值与标准差以支持抽样为统计单位，先平均训练重复。下列是固定图和checkpoint条件下的描述性结果，不能解释为跨图或独立预训练重复。", "",
        "F100固定历史配方，其他可训练方法支持内3折4候选选参。FrozenPrototypeCentered 使用与F100相同中心化原型读出；它是检验提示训练贡献的主要冻结对照。", ""]
    for job in plan["jobs"]:
        lines += ["## "+job["dataset"]+" / "+job["family"], "",
            "| 方法 | Accuracy | Macro-F1 | Micro-F1 | ROC-AUC | AP |",
            "|---|---:|---:|---:|---:|---:|"]
        for method in plan["methods"]:
            r = lookup[(job["id"], method)]
            cells = [f"{100*r[m+'_mean']:.3f} ± {100*r[m+'_std_over_support_draws']:.3f}" for m in METRICS]
            lines.append("| "+method+" | "+" | ".join(cells)+" |")
        lines += ["", "F100配对差值（百分点）及95% bootstrap描述性区间；全部参考方法如下。未做多重比较修正。", "",
                  "| 参考 | Accuracy差值 [95%区间] | Macro-F1差值 [95%区间] |", "|---|---:|---:|"]
        for method in plan["methods"][:-1]:
            cells = []
            for metric in ["accuracy", "macro_f1"]:
                p = next(p for p in pairs if (p["checkpoint_id"], p["reference"], p["metric"]) == (job["id"], method, metric))
                lo, hi = p["paired_bootstrap_95ci"]
                cells.append(f"{100*p['mean_delta']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}]")
            lines.append("| "+method+" | "+" | ".join(cells)+" |")
        lines += [""]
    lines += ["## 可主张的范围与验收边界", "",
        "PubMed用官方train/test，Actor与Chameleon用官方fold0；Photo与Computers按种子20260930每类20个节点建立候选支持池，其余节点为query。异配图保留加载后的原方向。全图特征和结构可见，只使用支持标签拟合，query评分在预测封存之后。", "",
        "原始权重的预训练种子未记录。缺少sidecar的Photo/Computers checkpoint为legacy_unverified；其他sidecar为ProG-adapted/not_fully_verified，不声称原作者代码逐位复现。", "",
        "GPF、GPFPlus、GraphPromptDiag是在统一GCN/同图少样本节点协议下的适配对照。F100有属性引导路由，但本轮未比较属性分类器，故不能声称优于属性方法或证明图结构的独立贡献。", "",
        f"正式coverage {len(seen)}/{len(expected)}，预测、状态、选参SHA、配对划分和独立五指标通过。保存分数重放完全相同；F100保存表示原型读出最大误差 {scientific['maximum_f100_serialized_readout_error']:.9g}，独立固定gate前向混合误差比例最大 {scientific['maximum_f100_mixed_forward_error_ratio']:.9g}（<1）。", "",
        f"完整新CUDA前向作为预声明诊断单独披露：全部方法最大概率误差 {scientific['maximum_independent_state_probability_error']:.9g}，类别变化 {scientific['independent_state_class_changes']}；F100重新构造gate全链最大概率误差 {scientific['f100_maximum_full_chain_probability_error']:.9g}，类别变化 {scientific['f100_full_chain_class_changes']}，严格<2e-6且类别一致标志为 {scientific['f100_strict_full_chain_passed']}。原始前向绝对<2e-6超限 {scientific['raw_f100_absolute_2e6_failures']} 单元；不把混合误差通过解释成逐位重放通过。", "",
        "逐单元重放诊断见各checkpoint的*_audit.json。五指标所有配对见paired_effects.json；源码与协议见plan.json与source/experiments/f100_multidata/PLAN.md。负结果、强冻结读出与微调差距完整保留，不据本轮query结果更改方法、模型、种子或统计口径。"]
    (output/"五数据集F100验证总结.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(scientific), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("workspace", type=Path)
    summarize(parser.parse_args().workspace)
