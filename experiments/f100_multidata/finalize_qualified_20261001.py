"""Deliver all fixed results while explicitly retaining failed numerical checks.

This does not change the original source, plan, predictions or acceptance limits.
Nine original checkpoint auditors passed. Actor/GPPT's collecting auditor checks
the whole matrix with the same threshold and records the seven failed units.
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def finalize(args):
    workspace = args.workspace; source_root = args.source_root
    sys.path.insert(0, str(source_root))
    import numpy as np
    import torch
    from prprompt.protocol import save_json, sha_file
    from experiments.f100_multidata.run import read_plan
    plan = read_plan(workspace/"plan.json")
    assert not list((workspace/"results").rglob("*.lock"))
    audits = []; failures = []
    for job in plan["jobs"]:
        original = workspace/"analysis"/(job["id"]+"_audit.json")
        collected = workspace/"analysis/numerical_collection"/original.name
        path = original if original.exists() else collected
        audit = json.loads(path.read_text())
        assert audit["present"] == audit["expected"] == 540 and not audit["smoke"]
        assert audit["plan_sha256"] == plan["plan_sha256"] and audit["missing"] == []
        if path == collected:
            assert audit["original_auditor_sha256"] == plan["source"]["experiments/f100_multidata/audit.py"]
            assert audit["collector_sha256"] == sha_file(workspace/"audit_failure_evidence/audit_collect_20261001.py")
            failures += audit["mixed_forward_failures"]
        audits.append(audit)
    assert len(audits) == 10 and sum(a["present"] for a in audits) == 5400
    assert len(failures) == 7, "Review a changed failure count, never assume numerical acceptance"
    assert all(f["mixed_forward_error_ratio"] >= 1 for f in failures)
    qualification = dict(predeclared_numerical_acceptance_passed=False,
        predeclared_mixed_forward_passed=False, mixed_forward_failure_count=len(failures), mixed_forward_failures=failures,
        original_failed_audit_preserved="audit_failure_evidence/formal_audit_7_0.log",
        original_frozen_auditor_sha256=plan["source"]["experiments/f100_multidata/audit.py"],
        collection_does_not_waive_failed_assertions=True,
        numerical_qualification="Seven Actor/GPPT F100 fresh forward comparisons exceed the original mixed bound. All source and sealed results retained. No training, threshold, seed, selection, scoring or statistics changes.")

    class RetainQualification(ast.NodeTransformer):
        changed = 0

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "audits":
                self.changed += 1
                node.value = ast.Name("complete_audits", ast.Load())
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace("（<1）。", "。预声明混合误差验收未全部通过，失败逐单元记录并保留。")
                node.value = node.value.replace("不把混合误差通过解释成逐位重放通过", "不以封存分数重放替代未通过的原始混合误差验收")
            return node

    summary_source = source_root/"experiments/f100_multidata/summarize.py"
    transform = RetainQualification(); tree = transform.visit(ast.parse(summary_source.read_text()))
    ast.fix_missing_locations(tree); assert transform.changed == 1
    namespace = dict(__file__=str(summary_source), __name__="qualified_summary", complete_audits=audits)
    exec(compile(tree, str(summary_source), "exec"), namespace)

    def save_qualified(path, value):
        if path.name == "scientific_audit.json":
            value.update(qualification)
        save_json(path, value)

    namespace["save_json"] = save_qualified
    namespace["summarize"](workspace)
    scientific = json.loads((workspace/"analysis/scientific_audit.json").read_text())
    assert not scientific["predeclared_numerical_acceptance_passed"]
    report = workspace/"analysis/五数据集F100验证总结.md"
    text = report.read_text()
    warning = ("**数值验收存在明确例外：Actor / Edgepred_GPPT 的7个F100单元未通过拟合前声明的混合前向误差界限。"
        "本报告保留全部5400份封存结果，且不将它们标为所有原数值测试均通过；封存预测、状态、选参完整性与五指标评分的验收单独记录。**\n\n")
    diagnostic = json.loads((workspace/"audit_failure_evidence/repeated_forward_diagnostic.json").read_text())
    warning += (f"固定第一个失败单元在两张GPU各重复10次，无拟合、无query标签、无丢弃重放；"
        f"原始表示最大重复抖动 {diagnostic['maximum_raw_repeat_jitter']:.9g}，最大概率误差 {diagnostic['maximum_probability_error']:.9g}，"
        f"重放类别变化 {diagnostic['classification_changes']}。这与CUDA float32稀疏归约数值差异相符，是定位证据，不是将原失败改判通过的依据。\n\n")
    text = text.replace("# 未修正 Edgepred＋v1 F100：五个新增数据集\n\n", "# 未修正 Edgepred＋v1 F100：五个新增数据集\n\n"+warning, 1)
    table = json.loads((workspace/"analysis/method_comparison.json").read_text())
    pairs = json.loads((workspace/"analysis/paired_effects.json").read_text())
    lookup = {(r["checkpoint_id"], r["method"]): r for r in table}
    overview = ["## 效果概览", "", "Accuracy均值（%）；统计单位和全部方法五指标见后文。", "",
        "| 数据集 | 原权重 | v1 F100 | 冻结中心化原型 | FullFineTune | v2 PRPrompt |",
        "|---|---|---:|---:|---:|---:|"]
    for job in plan["jobs"]:
        cells = [f"{100*lookup[(job['id'], method)]['accuracy_mean']:.3f}" for method in ["F100", "FrozenPrototypeCentered", "FullFineTune", "PRPrompt"]]
        overview.append("| "+job["dataset"]+" | "+job["family"]+" | "+" | ".join(cells)+" |")
    overview += [""]
    for reference, label in [("FrozenPrototypeCentered", "冻结中心化原型"), ("FullFineTune", "完整微调"), ("PRPrompt", "v2 PRPrompt")]:
        selected = [p for p in pairs if p["reference"] == reference and p["metric"] == "accuracy"]
        assert len(selected) == 10
        positive = sum(p["mean_delta"] > 0 for p in selected)
        positive_ci = sum(p["paired_bootstrap_95ci"][0] > 0 for p in selected)
        negative_ci = sum(p["paired_bootstrap_95ci"][1] < 0 for p in selected)
        overview.append(f"相对{label}，F100的Accuracy均值在 {positive}/10 个条件更高；配对描述性区间完全为正 {positive_ci}/10，完全为负 {negative_ci}/10。")
    overview += ["", "上述覆盖计数不替代逐条件判断；Actor/GPPT保留数值验收例外，未知预训练种子、固定图/划分及支持内选参差异限制推广范围。", ""]
    index = text.index("## "+plan["jobs"][0]["dataset"]+" / ")
    text = text[:index]+"\n".join(overview)+"\n"+text[index:]
    text += ("\n## 未通过测试及源码保存\n\n原审计源码和原plan.json完整保留。补充采集器只将单个mixed-forward断言延期至记录完成，"
        "阈值与判定未改变，其他断言继续执行。九个checkpoint原审计通过；Actor/GPPT全部540个方法单元由采集器检查，其中60个F100单元的7个原混合误差测试失败。"
        "该条件封存表示读出最大误差0，固定路由重放概率最大误差约4.03e-7，无类别变化；独立重建gate全链概率约2.95e-6超出原2e-6诊断阈值，也如实保存。\n\n"
        "详细记录位于analysis/numerical_collection/与audit_failure_evidence/，全部改动仅在附加审计、失败信息传播和结果交付，不重新拟合或更改正式实验源码。"
        "scientific_audit.json和local_delivery_status.json分别保留原数值验收False及交付/评分通过状态。\n")
    report.write_text(text)
    provenance = []
    for dataset, spec in plan["datasets"].items():
        data = torch.load(spec["snapshot"], map_location="cpu", weights_only=True)
        target = workspace/"audit_reference"/(dataset+".npz"); target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target, labels=data["labels"].numpy(), pool=data["pool"].numpy(), query=data["query"].numpy())
        provenance.append(dict(dataset=dataset, file=target.name, sha256=sha_file(target),
            graph_sha256=spec["graph_sha256"], labels_sha256=spec["labels_sha256"], exported_after_sealing=True))
    save_json(workspace/"audit_reference/provenance.json", provenance)
    with (workspace/"environment.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    save_json(workspace/"audit_failure_evidence/finalization_provenance.json", dict(
        finalizer_sha256=sha_file(Path(__file__)), original_summary_sha256=sha_file(summary_source),
        original_plan_sha256=plan["plan_sha256"], scientific_fit_source_changed=False,
        training_repeated=False, original_numerical_limits_changed=False, original_numerical_acceptance_passed=False))
    original_status = json.loads((workspace/"audit_failure_evidence/pipeline_status.json").read_text())
    save_json(workspace/"pipeline_status.json", dict(pid=os.getpid(), phase="server_complete", active=[], pending=0,
        plan_sha256=plan["plan_sha256"], formal_units=5400, finished_at=time.time(),
        completion_class="delivery_ready_with_numerical_qualification", numerical_acceptance_passed=False,
        initial_pipeline_failure=original_status["error"], mixed_forward_failure_count=len(failures),
        next_action="Complete local SHA, NumPy score/summary/pair audit; retain numerical failure in final report"))
    (workspace/"logs/controller.log").write_bytes((workspace/"pipeline.log").read_bytes())
    manifest = {}
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if not path.is_file() or path.is_symlink() or path.name == "result_files_sha256.json":
            continue
        if relative.parts[0] == "data" or "__pycache__" in relative.parts or path.name in ["pipeline.log", "pipeline.pid"]:
            continue
        assert path.suffix != ".lock", path
        manifest[str(relative)] = sha_file(path)
    save_json(workspace/"result_files_sha256.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    finalize(parser.parse_args())
