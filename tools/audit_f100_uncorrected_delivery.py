"""Portable uncorrected F100 delivery audit, NumPy-only."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from metric_utils import numpy_metrics, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root/"result_files_sha256.json").read_text())
    for name, expected in manifest.items():
        assert sha(root/name) == expected, name
    labels, queries, graphs = {}, {}, {}
    for row in json.loads((root/"audit_reference/provenance.json").read_text()):
        file = root/"audit_reference"/row["file"]
        assert sha(file) == row["sha256"]
        with np.load(file) as data:
            labels[row["dataset"]] = data["labels"].copy()
            queries[row["dataset"]] = data["query"].copy()
        graphs[row["dataset"]] = row["graph_sha256"]
    plan = json.loads((root/"plan.json").read_text())
    paths = list((root/"results").glob("*/5/*/*/F100/metrics.json"))
    assert len(paths) == plan["expected_units"] == 240
    expected = {(j["id"], s, t) for j in plan["jobs"] for s in plan["split_seeds"] for t in plan["train_seeds"]}
    seen = set()
    error_max = aggregate_error = 0.
    blocks = defaultdict(lambda: defaultdict(list))
    for path in paths:
        row = json.loads(path.read_text())
        identity = (row["checkpoint_id"], row["split_seed"], row["train_seed"])
        assert identity in expected and identity not in seen
        seen.add(identity)
        assert row["plan_sha256"] == plan["plan_sha256"] and row["graph_sha256"] == graphs[row["dataset"]]
        assert not row["smoke"]
        for name, key in [("state.pt", "state_sha256"), ("selection.json", "selection_sha256"), ("predictions.npz", "predictions_sha256")]:
            assert sha(path.parent/name) == row[key]
        seal = json.loads((path.parent/"seal.json").read_text())
        assert seal["predictions_sha256"] == row["predictions_sha256"] and seal["signature"] == row["signature"]
        with np.load(path.parent/"predictions.npz") as data:
            p, q, s = data["probabilities"], data["query"], data["support"]
            assert np.array_equal(q, queries[row["dataset"]]) and not np.isin(s, q).any()
            assert np.all(np.bincount(labels[row["dataset"]][s], minlength=p.shape[1]) == 5)
            got = numpy_metrics(labels[row["dataset"]][q], p)
        for metric, value in got.items():
            error = abs(value-row["metrics"][metric])
            error_max = max(error_max, error)
            assert error < 1e-12
        blocks[row["checkpoint_id"]][row["split_seed"]].append(got)
    assert seen == expected
    with (root/"analysis/checkpoint_comparison.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["method"] == "v1_F100"]
    assert len(rows) == 4
    for row in rows:
        draws = blocks[row["checkpoint_id"]]
        assert len(draws) == 20 and all(len(reps) == 3 for reps in draws.values())
        for metric in next(iter(next(iter(draws.values())))):
            values = [np.mean([r[metric] for r in reps]) for reps in draws.values()]
            for actual, key in [(np.mean(values), metric+"_mean"), (np.std(values, ddof=1), metric+"_std_over_support_draws")]:
                error = abs(float(actual)-float(row[key]))
                aggregate_error = max(aggregate_error, error)
                assert error < 1e-12
    report = {"phase": "complete", "manifest_files_checked": len(manifest), "sha256_mismatches": 0,
              "formal_units": len(paths), "missing": [], "five_metrics_recomputed_locally": True,
              "maximum_local_metric_error": error_max, "maximum_local_aggregate_error": aggregate_error,
              "support_query_isolation_passed": True, "checkpoint_rows_reaggregated": 4,
              "report": "analysis/未修正Edgepred_F100验证总结.md", "report_sha256": sha(root/"analysis/未修正Edgepred_F100验证总结.md")}
    (root/"local_delivery_status.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
