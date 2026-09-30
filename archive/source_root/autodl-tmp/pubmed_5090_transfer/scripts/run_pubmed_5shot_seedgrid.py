#!/usr/bin/env python
"""PubMed 5-shot seed-grid matrix under the v4 protocol (single GPU).

Mirrors ``run_cora_5shot_seedgrid.py`` but for the PubMed dataset and pinned
to one GPU. Each of the 241 cells (24 prompts x 10 pretrains + MultiGprompt) is
run under a 3x3 ``split_seed`` x ``train_seed`` grid -> 2169 runs. PubMed
checkpoints were retrained with provenance sidecars first (see preamble).

Usage:
    python scripts/run_pubmed_5shot_seedgrid.py [--gpu 0] [--dry-run]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_cora_5shot_matrix as M  # reuse helpers  # noqa: E402

DATASET = "PubMed"
REPO_ROOT = M.REPO_ROOT
CKPT_ROOT = REPO_ROOT / "Experiment" / "pre_trained_model" / DATASET
OUT_CSV = REPO_ROOT / "results" / f"pubmed_5shot_full_matrix_seedgrid.csv"
CELL_DIR = REPO_ROOT / "results" / "pubmed_5shot_seedgrid"
ARTIFACT_ROOT = CELL_DIR / "artifacts"
LOG_DIR = REPO_ROOT / "scripts" / "baseline_logs" / "pubmed_5shot_seedgrid"

SHOT = 5
SPLIT_SEEDS = [1729, 17290, 172900]
TRAIN_SEEDS = [42, 4242, 424242]
SEED_PAIRS = list(product(SPLIT_SEEDS, TRAIN_SEEDS))  # 9


def ckpt_path(pretrain: str) -> str:
    if pretrain == "None":
        return "None"
    if pretrain == "MultiGprompt":
        return str(CKPT_ROOT / "MultiGprompt.pth")
    return str(CKPT_ROOT / f"{pretrain}.GCN.128hidden_dim.pth")


def seedgrid_key(prompt: str, pretrain: str, split_seed: int, train_seed: int) -> str:
    return f"{M.cell_key(prompt, pretrain)}_s{split_seed}_t{train_seed}"


def build_config(prompt: str, pretrain: str, split_seed: int, train_seed: int, device: int) -> dict:
    config = dict(
        dataset=DATASET,
        prompt=prompt,
        checkpoint=ckpt_path(pretrain),
        gnn="GCN",
        hidden_dim=128,
        layers=2,
        shot=SHOT,
        folds=5,
        support_passes=200,
        learning_rate=1e-3,
        weight_decay=1e-5,
        split_seed=split_seed,
        train_seed=train_seed,
        inference_seed=train_seed + 1000000,
        device=str(device),
        subgraph_min=100,
        subgraph_max=300,
        output_root=str(ARTIFACT_ROOT),
        split_root=str(M.SPLIT_ROOT),
    )
    if prompt == "GPPT":
        config["learning_rate"] = 2e-3
        config["weight_decay"] = 5e-4
    if prompt == "All-in-one":
        config["answer_passes"] = 50
        config["prompt_passes"] = 50
        config["prompt_lr"] = 1e-6
    return config


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
    config = build_config(prompt, pretrain, split_seed, train_seed, device)
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
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=_proc_env())

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
            dataset=result.get("dataset", DATASET),
            shot=SHOT,
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
            status="skipped", dataset=DATASET, shot=SHOT, prompt=prompt, pretrain=pretrain,
            split_seed=split_seed, train_seed=train_seed, device=str(device),
            reason=status.get("reason")))
        return 0

    M.write_json(index_path, dict(
        status="failed", dataset=DATASET, shot=SHOT, prompt=prompt, pretrain=pretrain,
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
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent processes, all pinned to --gpu (OMP/MKL=2)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("workers must be positive")

    CELL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cells = build_seedgrid_cells()
    print(
        f"PubMed seedgrid runs: {len(cells)} (cells={len(M.build_cells())} x "
        f"seed_pairs={len(SEED_PAIRS)}, gpu={args.gpu}, workers={args.workers})",
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
        for prompt, pretrain, split_seed, train_seed in todo:
            print(f"  {SHOT}shot {pretrain}+{prompt} s={split_seed} t={train_seed} -> gpu {args.gpu}")
        return 0

    done = 0
    failed = 0

    def run_one(item):
        prompt, pretrain, split_seed, train_seed = item
        rc = run_seedgrid_cell(prompt, pretrain, split_seed, train_seed, args.gpu)
        return item, rc

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for item, rc in ex.map(run_one, todo):
            prompt, pretrain, split_seed, train_seed = item
            if rc == 0:
                done += 1
            else:
                failed += 1
            print(f"{pretrain}+{prompt} s={split_seed} t={train_seed} rc={rc}", flush=True)

    n = export_csv()
    print(f"done={done} failed={failed} csv_rows={n}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
