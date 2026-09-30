#!/usr/bin/env python
"""Cora 5-shot seed-grid matrix under the v4 protocol.

Same cell population as ``run_cora_5shot_matrix.py`` (241 cells), but each cell
is re-run under a 3x3 grid of ``split_seed`` x ``train_seed`` pairs, giving
241 x 9 = 2169 distinct runs. Distinct seed pairs produce distinct content-
addressed artifacts automatically (both seeds feed the v4 run_id), so no engine
change is required: this scheduler only drives ``run_protocol_cell.py --config``
with the seed variants and records a per-cell index.

Concurrency: spawned subprocesses are CPU-bound on the induced-subgraph build
and are thread-oversubscribed by default (super-linear slowdown). Pinning each
subprocess to ``OMP_NUM_THREADS=2``/``MKL_NUM_THREADS=2`` restores near-linear
scaling; ``--workers`` processes are spread round-robin across the 2 GPUs via
``device = i % 2``.

Usage:
    python scripts/run_cora_5shot_seedgrid.py [--workers 4] [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

# Reuse the single-seed matrix's helpers (cells, config builder, metric mapping).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_cora_5shot_matrix as M  # noqa: E402

REPO_ROOT = M.REPO_ROOT
OUT_CSV = REPO_ROOT / "results" / "cora_5shot_full_matrix_seedgrid.csv"
CELL_DIR = REPO_ROOT / "results" / "cora_5shot_seedgrid"
ARTIFACT_ROOT = CELL_DIR / "artifacts"
LOG_DIR = REPO_ROOT / "scripts" / "baseline_logs" / "cora_5shot_seedgrid"

SPLIT_SEEDS = [1729, 17290, 172900]
TRAIN_SEEDS = [42, 4242, 424242]
SEED_PAIRS = list(product(SPLIT_SEEDS, TRAIN_SEEDS))  # 9 pairs


def seedgrid_key(prompt: str, pretrain: str, split_seed: int, train_seed: int) -> str:
    return f"{M.cell_key(prompt, pretrain)}_s{split_seed}_t{train_seed}"


def build_seedgrid_cells() -> list[tuple[str, str, int, int]]:
    return [
        (prompt, pretrain, split_seed, train_seed)
        for prompt, pretrain in M.build_cells()
        for split_seed, train_seed in SEED_PAIRS
    ]


def _proc_env() -> dict:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    return env


def run_seedgrid_cell(prompt: str, pretrain: str, split_seed: int, train_seed: int, device: int) -> int:
    key = seedgrid_key(prompt, pretrain, split_seed, train_seed)
    config = M.build_config(prompt, pretrain, device)
    config["split_seed"] = split_seed
    config["train_seed"] = train_seed
    if "inference_seed" in config:
        del config["inference_seed"]  # let the engine derive from train_seed
    config["output_root"] = str(ARTIFACT_ROOT)
    config_path = CELL_DIR / f"{key}.config.json"
    index_path = CELL_DIR / f"{key}.json"
    M.write_json(config_path, config)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_protocol_cell.py"),
        "--config",
        str(config_path),
    ]
    log_path = LOG_DIR / f"{key}.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=_proc_env()
        )

    status_line = ""
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            status_line = line
    try:
        status = json.loads(status_line) if status_line else {}
    except json.JSONDecodeError:
        status = {}

    status_name = status.get("status")
    if status_name == "complete":
        artifact = Path(status["output_directory"])
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        index = dict(
            status="complete",
            dataset=result.get("dataset", "Cora"),
            shot=M.SHOT,
            prompt=prompt,
            pretrain=pretrain,
            split_seed=split_seed,
            train_seed=train_seed,
            device=str(device),
            protocol=result.get("protocol"),
            run_id=result.get("run_id"),
            artifact_directory=str(artifact),
            variant=result.get("variant"),
            input_mode=result.get("input_mode"),
            readout=result.get("readout"),
            head=result.get("head"),
            checkpoint_provenance=result.get("checkpoint_provenance"),
        )
        M.write_json(index_path, index)
        return 0

    if status_name == "skipped":
        M.write_json(index_path, dict(
            status="skipped", dataset="Cora", shot=M.SHOT, prompt=prompt, pretrain=pretrain,
            split_seed=split_seed, train_seed=train_seed, device=str(device),
            reason=status.get("reason")))
        return 0

    M.write_json(index_path, dict(
        status="failed", dataset="Cora", shot=M.SHOT, prompt=prompt, pretrain=pretrain,
        split_seed=split_seed, train_seed=train_seed, device=str(device),
        reason=status.get("reason"), exit_code=proc.returncode))
    return 1


def export_csv() -> int:
    import pandas as pd

    rows = []
    for prompt, pretrain, split_seed, train_seed in build_seedgrid_cells():
        index_path = CELL_DIR / f"{seedgrid_key(prompt, pretrain, split_seed, train_seed)}.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("status") != "complete":
            rows.append(index)
            continue
        result = json.loads(
            (Path(index["artifact_directory"]) / "result.json").read_text(encoding="utf-8")
        )
        m = M.metrics_from_result(result)
        row = {
            "dataset": index.get("dataset"),
            "shot": index.get("shot"),
            "prompt": index.get("prompt"),
            "pretrain": index.get("pretrain"),
            "split_seed": index.get("split_seed"),
            "train_seed": index.get("train_seed"),
            "device": index.get("device"),
            "status": "complete",
            "protocol": index.get("protocol"),
            "run_id": index.get("run_id"),
            "artifact_directory": index.get("artifact_directory"),
            "variant": index.get("variant"),
            "input_mode": index.get("input_mode"),
            "readout": index.get("readout"),
            "head": index.get("head"),
            "checkpoint_provenance": index.get("checkpoint_provenance"),
        }
        for name, alias in M.METRIC_ALIASES.items():
            row[f"{alias}_mean"] = m[f"{name}_mean"]
            row[f"{alias}_std"] = m[f"{name}_std"]
        rows.append(row)

    if not rows:
        print("no results to export", flush=True)
        return 0

    df = pd.DataFrame(rows)
    pt_order = {pt: i for i, pt in enumerate(M.PRETRAINS + ["MultiGprompt"])}
    pr_order = {p: i for i, p in enumerate(M.PROMPTS + ["MultiGprompt"])}
    ss_order = {s: i for i, s in enumerate(SPLIT_SEEDS)}
    ts_order = {s: i for i, s in enumerate(TRAIN_SEEDS)}
    df["_pt"] = df["pretrain"].map(pt_order)
    df["_pr"] = df["prompt"].map(pr_order)
    df["_ss"] = df["split_seed"].map(ss_order)
    df["_ts"] = df["train_seed"].map(ts_order)
    df = df.sort_values(["shot", "_pt", "_pr", "_ss", "_ts"]).drop(columns=["_pt", "_pr", "_ss", "_ts"])
    cols = [
        "dataset", "shot", "prompt", "pretrain", "split_seed", "train_seed", "device",
        "status", "protocol", "run_id", "artifact_directory", "variant", "input_mode",
        "readout", "head", "checkpoint_provenance",
        "accuracy_mean", "accuracy_std", "f1_mean", "f1_std",
        "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)", flush=True)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("workers must be positive")

    CELL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cells = build_seedgrid_cells()
    print(
        f"total seedgrid runs: {len(cells)} (cells={len(M.build_cells())} x "
        f"seed_pairs={len(SEED_PAIRS)}, workers={args.workers})",
        flush=True,
    )

    todo = []
    for (prompt, pretrain, split_seed, train_seed) in cells:
        index_path = CELL_DIR / f"{seedgrid_key(prompt, pretrain, split_seed, train_seed)}.json"
        if index_path.exists():
            try:
                existing = json.loads(index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if existing.get("status") in {"complete", "skipped"}:
                continue
        todo.append((prompt, pretrain, split_seed, train_seed))
    print(f"already done: {len(cells) - len(todo)}; remaining: {len(todo)}", flush=True)

    if args.dry_run:
        for i, (prompt, pretrain, split_seed, train_seed) in enumerate(todo):
            print(f"  {M.SHOT}shot {pretrain}+{prompt} s={split_seed} t={train_seed} -> gpu {i % 2}")
        return 0

    if not todo:
        return export_csv()

    done_count = 0
    fail_count = 0
    failed: list[str] = []

    def run_one(item):
        prompt, pretrain, split_seed, train_seed = item
        # Device assigned by stable global index so run_id is reproducible.
        i = cells.index(item)
        device = i % 2
        rc = run_seedgrid_cell(prompt, pretrain, split_seed, train_seed, device)
        return item, rc

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for item, rc in ex.map(run_one, todo):
            prompt, pretrain, split_seed, train_seed = item
            if rc == 0:
                done_count += 1
            else:
                fail_count += 1
                failed.append(f"{pretrain}+{prompt}(s{split_seed},t{train_seed})")
            print(f"{pretrain}+{prompt} s={split_seed} t={train_seed} rc={rc}", flush=True)

    n = export_csv()
    print(f"done_count={done_count} fail_count={fail_count} csv_rows={n}", flush=True)
    if fail_count:
        print("failed cells:", ", ".join(failed), flush=True)
        print("re-run this script to retry failed cells", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
