"""Verify original-weight F100 comparisons against immutable delivered references."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from metric_utils import sha

METRICS = ["accuracy", "macro_f1", "micro_f1", "roc_auc_macro", "average_precision_macro"]
MAIN = ["FrozenLinear", "FrozenPrototype", "FullFineTune", "GPF", "GPFPlus", "GraphPromptDiag", "PRPrompt"]
PAIRS = ["PRPrompt", "FullFineTune"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    plan = json.loads((root/"plan.json").read_text())
    roots = {"main": root/"references/uncorrected_main"}
    manifests = {}
    for name, directory in roots.items():
        reference = plan["references"][name]
        file = directory/Path(reference["manifest_file"]).relative_to(reference["root"])
        assert sha(file) == reference["manifest_file_sha256"]
        planfile = directory/Path(reference["plan_file"]).relative_to(reference["root"])
        assert sha(planfile) == reference["plan_file_sha256"]
        manifests[name] = json.loads(file.read_text())
    base = roots["main"]/Path(plan["base"]).relative_to(plan["references"]["main"]["root"])
    blocks = defaultdict(lambda: defaultdict(list))
    checked = 0
    for path in sorted((root/"results").glob("*/5/*/*/F100/metrics.json")):
        row = json.loads(path.read_text())
        identity = row["checkpoint_id"]
        blocks[(identity, "v1_F100")][row["split_seed"]].append(row["metrics"])
        relative = Path(identity)/"5"/str(row["split_seed"])/str(row["train_seed"])
        sources = [(m, base/relative/m/"metrics.json", "main") for m in MAIN]
        for method, file, name in sources:
            assert sha(file) == manifests[name][str(file.relative_to(roots[name]))]
            ref = json.loads(file.read_text())
            for key in ["support_sha256", "query_sha256", "support_labels_sha256", "graph_sha256"]:
                assert ref[key] == row[key]
            blocks[(identity, method)][row["split_seed"]].append(ref["metrics"])
            checked += 1
    means = {}
    for key, draws in blocks.items():
        reps_expected = 3
        assert len(draws) == 20 and all(len(reps) == reps_expected for reps in draws.values())
        means[key] = {s: {m: float(np.mean([r[m] for r in reps])) for m in METRICS} for s, reps in draws.items()}
    aggregate_error = paired_error = 0.
    with (root/"analysis/checkpoint_comparison.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 32
    for row in table:
        draws = means[(row["checkpoint_id"], row["method"])]
        for m in METRICS:
            values = [r[m] for r in draws.values()]
            for actual, key in [(np.mean(values), m+"_mean"), (np.std(values, ddof=1), m+"_std_over_support_draws")]:
                error = abs(float(actual)-float(row[key])); aggregate_error = max(aggregate_error, error)
                assert error < 1e-12
    pairs = json.loads((root/"analysis/paired_effects.json").read_text())
    lookup = {(r["checkpoint_id"], r["reference"], r["metric"]): r for r in pairs}
    assert len(lookup) == len(pairs) == 40
    rng = np.random.default_rng(plan["seed_for_bootstrap"])
    for j in plan["jobs"]:
        target = means[(j["id"], "v1_F100")]
        for reference in PAIRS:
            baseline = means[(j["id"], reference)]
            for m in METRICS:
                delta = np.array([target[s][m]-baseline[s][m] for s in sorted(target)])
                ci = np.quantile(rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1), [.025, .975])
                row = lookup[(j["id"], reference, m)]
                error = max(abs(float(delta.mean())-row["mean_delta"]), float(np.max(np.abs(ci-row["paired_bootstrap_95ci"]))))
                paired_error = max(paired_error, error)
                assert error < 1e-12
        rng.integers(0, len(target), size=(5, 10000, len(target)))
    scientific = json.loads((root/"analysis/scientific_audit.json").read_text())
    report = {"reference_files_checked": checked, "reference_sha256_mismatches": 0,
              "all_reference_graph_support_query_hashes_match": True, "checkpoint_method_rows_reaggregated": 32,
              "paired_effects_and_intervals_recomputed": 40, "maximum_aggregate_error": aggregate_error,
              "maximum_paired_effect_or_interval_error": paired_error,
              "original_full_chain_replay_passed": scientific["original_full_chain_replay_passed"],
              "serialized_state_replay_passed": scientific["replay_argmax_equal"] and scientific["maximum_final_probability_replay_error"]<2e-6}
    (root/"local_paired_reference_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    status = json.loads((root/"local_delivery_status.json").read_text())
    status.update(paired_reference_audit=report)
    (root/"local_delivery_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
