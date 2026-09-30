"""Validate completeness and produce method comparisons, never factorial labels."""
import argparse
import csv
import itertools
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from prprompt.protocol import sha_file, save_json


def summarize(root, allow_partial=False):
    plan = json.loads((root / "run_plan.json").read_text())
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    actual_plan_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if actual_plan_hash != plan["plan_sha256"]:
        raise ValueError("Run plan changed after execution")
    expected = {(j["id"], k, s, t, m) for j, k, s, t, m in itertools.product(
        plan["jobs"], plan["config"]["shots"], plan["config"]["split_seeds"],
        plan["config"]["train_seeds"], plan["config"]["methods"])}
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    pairing = {}
    for path in sorted(root.glob("*/*/*/*/*/metrics.json")):
        row = json.loads(path.read_text())
        identity = tuple(row[k] for k in ["checkpoint_id", "shots", "split_seed", "train_seed", "method"])
        if identity in seen or identity not in expected or row["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError(f"Unexpected/duplicate run: {path}")
        if sha_file(path.parent / "predictions.npz") != row["predictions_sha256"]:
            raise ValueError(f"Changed predictions: {path}")
        for filename, field in [("state.pt", "state_sha256"), ("selection.json", "selection_sha256")]:
            if sha_file(path.parent / filename) != row[field]:
                raise ValueError(f"Changed {filename}: {path}")
        seal = json.loads((path.parent / "seal.json").read_text())
        for field in ["plan_sha256", "predictions_sha256", "support_sha256", "query_sha256", "signature"]:
            if seal[field] != row[field]:
                raise ValueError(f"Seal/report mismatch: {path}")
        pair_key = (row["checkpoint_id"], row["shots"], row["split_seed"])
        pair_value = tuple(row[k] for k in ["graph_sha256", "split_policy_sha256", "support_sha256", "query_sha256", "support_labels_sha256", "evaluation_labels_sha256"])
        if pair_key in pairing and pairing[pair_key] != pair_value:
            raise ValueError(f"Methods do not share a matched split: {path}")
        pairing[pair_key] = pair_value
        seen.add(identity)
        key = (row["dataset"], row["family"], row["checkpoint_id"], row["shots"], row["method"])
        groups[key][row["split_seed"]].append(row)
    missing = expected - seen
    save_json(root / "coverage.json", {"expected": len(expected), "present": len(seen), "missing": sorted(missing), "smoke": plan["smoke"]})
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} runs missing; use --allow-partial only for progress, not a paper table")
    table = []
    metrics = ["accuracy", "macro_f1", "micro_f1", "roc_auc_macro", "average_precision_macro"]
    block_values = {}
    for key, blocks in sorted(groups.items()):
        row = dict(zip(["dataset", "family", "checkpoint_id", "shots", "method"], key))
        row.update(support_draws=len(blocks), training_runs=sum(map(len, blocks.values())), smoke=plan["smoke"], incomplete=bool(missing))
        for metric in metrics:
            values = [np.mean([r["metrics"][metric] for r in reps]) for reps in blocks.values()
                      if all(r["metrics"][metric] is not None for r in reps)]
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_std_over_support_draws"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row["trainable_parameters"] = next(iter(blocks.values()))[0]["audit"]["trainable_parameters"]
        row["total_seconds_mean"] = float(np.mean([r["total_seconds"] for reps in blocks.values() for r in reps]))
        block_values[key] = {seed: np.mean([r["metrics"]["accuracy"] for r in reps]) for seed, reps in blocks.items()}
        table.append(row)
    if table:
        with (root / "method_comparison.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    paired = []
    rng = np.random.default_rng(20260930)
    for key, ours in block_values.items():
        if key[-1] != "PRPrompt":
            continue
        for other, values in block_values.items():
            if key[:-1] != other[:-1] or other[-1] == "PRPrompt":
                continue
            shared = sorted(set(ours) & set(values))
            delta = np.array([ours[s] - values[s] for s in shared])
            ci = np.quantile(rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1), [0.025, 0.975]).tolist() if len(delta) >= 2 else None
            paired.append({"checkpoint_id": key[2], "shots": key[3], "baseline": other[-1],
                           "support_draws": len(delta), "accuracy_delta": float(delta.mean()), "paired_bootstrap_95ci": ci})
    save_json(root / "paired_comparisons.json", {"smoke": plan["smoke"], "incomplete": bool(missing),
              "unit": "support draw; training seeds averaged first; one graph/checkpoint per stratum", "rows": paired})
    label = "SMOKE — NOT PAPER RESULTS" if plan["smoke"] else ("INCOMPLETE — NOT PAPER RESULTS" if missing else "Method comparison")
    lines = [f"# {label}", "", "Values are percentages; variability is across support draws after averaging training seeds.", "",
             "| Dataset | Checkpoint | Shots | Method | Accuracy | Macro F1 | Trainable parameters |", "|---|---|---:|---|---:|---:|---:|"]
    for r in table:
        def cell(metric):
            mean, std = r[metric + "_mean"], r[metric + "_std_over_support_draws"]
            return f"{100*mean:.3f}" + (f" ± {100*std:.3f}" if std is not None else " (one draw)")
        lines.append(f"| {r['dataset']} | {r['checkpoint_id']} | {r['shots']} | {r['method']} | {cell('accuracy')} | {cell('macro_f1')} | {r['trainable_parameters']} |")
    (root / "method_comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"expected": len(expected), "present": len(seen), "rows": len(table), "smoke": plan["smoke"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_directory", type=Path)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    summarize(args.run_directory, args.allow_partial)
