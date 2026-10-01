"""Local complete SHA, independent NumPy metrics, aggregation and pairing audit."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from metric_utils import sha, numpy_metrics
METRICS = ["accuracy", "macro_f1", "micro_f1", "roc_auc_macro", "average_precision_macro"]


def main(root):
    manifest = json.loads((root/"result_files_sha256.json").read_text())
    for name, value in manifest.items():
        assert sha(root/name) == value, name
    plan = json.loads((root/"plan.json").read_text())
    identity_key = "document_sha256" if "document_sha256" in plan else "plan_sha256"
    payload = {k: v for k, v in plan.items() if k != identity_key}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == plan[identity_key]
    for name, value in plan["source"].items():
        assert sha(root/"source"/name) == value
    labels = {}; queries = {}
    for dataset in plan["datasets"]:
        with np.load(root/"audit_reference"/(dataset+".npz")) as data:
            labels[dataset] = data["labels"].copy(); queries[dataset] = data["query"].copy()
    expected = {(j["id"], s, t, m) for j in plan["jobs"] for s in plan["split_seeds"]
                for t in plan["train_seeds"] for m in plan["methods"]}
    seen = set(); groups = collections.defaultdict(lambda: collections.defaultdict(list)); max_error = 0.
    for path in sorted((root/"results").glob("*/5/*/*/*/metrics.json")):
        row = json.loads(path.read_text())
        key = tuple(row[k] for k in ["checkpoint_id", "split_seed", "train_seed", "method"])
        assert key in expected and key not in seen; seen.add(key)
        assert not row["smoke"] and row["plan_sha256"] == plan["plan_sha256"]
        for name, field in [("predictions.npz", "predictions_sha256"), ("state.pt", "state_sha256"), ("selection.json", "selection_sha256")]:
            assert sha(path.parent/name) == row[field]
        with np.load(path.parent/"predictions.npz") as data:
            p, q, s = data["probabilities"], data["query"], data["support"]
            assert np.array_equal(q, queries[row["dataset"]]) and not np.isin(s, q).any()
            assert len(np.unique(s)) == len(s)
            assert np.all(np.bincount(labels[row["dataset"]][s], minlength=p.shape[1]) == 5)
            values = numpy_metrics(labels[row["dataset"]][q], p)
        for metric, value in values.items():
            error = abs(value-row["metrics"][metric]); max_error = max(max_error, error); assert error < 1e-12
        groups[(row["checkpoint_id"], row["method"])][row["split_seed"]].append(values)
    assert seen == expected and len(seen) == plan["expected_units"]
    blocks = {}; summary_error = 0.
    for row in json.loads((root/"analysis/method_comparison.json").read_text()):
        draws = groups[(row["checkpoint_id"], row["method"])]
        assert len(draws) == 20 and all(len(reps) == 3 for reps in draws.values())
        values = {s: {m: float(np.mean([r[m] for r in reps])) for m in METRICS} for s, reps in draws.items()}
        blocks[(row["checkpoint_id"], row["method"])] = values
        for metric in METRICS:
            v = [r[metric] for r in values.values()]
            for suffix, actual in [("_mean", np.mean(v)), ("_std_over_support_draws", np.std(v, ddof=1))]:
                error = abs(actual-row[metric+suffix]); summary_error = max(summary_error, error); assert error < 1e-12
    pairs = json.loads((root/"analysis/paired_effects.json").read_text()); index = 0; pair_error = 0.
    rng = np.random.default_rng(plan["bootstrap_seed"])
    for job in plan["jobs"]:
        for method in plan["methods"]:
            if method == "F100":
                continue
            for metric in METRICS:
                row = pairs[index]; index += 1
                assert (row["checkpoint_id"], row["reference"], row["metric"]) == (job["id"], method, metric)
                ours, other = blocks[(job["id"], "F100")], blocks[(job["id"], method)]
                delta = np.array([ours[s][metric]-other[s][metric] for s in sorted(ours)])
                ci = np.quantile(rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1), [.025, .975])
                errors = [abs(float(delta.mean())-row["mean_delta"]), *np.abs(ci-row["paired_bootstrap_95ci"])]
                pair_error = max(pair_error, *errors); assert max(errors) < 1e-12
    assert index == len(pairs) == 400 and len(blocks) == 90
    audit = json.loads((root/"analysis/scientific_audit.json").read_text())
    assert audit["present"] == audit["expected"] == len(seen) and audit["missing"] == []
    status = dict(phase="complete", plan_sha256=plan["plan_sha256"], files_verified=len(manifest), sha256_mismatches=0,
        formal_units=len(seen), checkpoint_method_rows=90, paired_metric_rows=400,
        five_metrics_independently_recomputed_locally=True, maximum_local_metric_error=max_error,
        maximum_summary_error=summary_error, maximum_paired_bootstrap_error=pair_error,
        f100_strict_full_chain_passed=audit["f100_strict_full_chain_passed"],
        report="analysis/五数据集F100验证总结.md", report_sha256=sha(root/"analysis/五数据集F100验证总结.md"))
    (root/"local_delivery_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory.resolve())
