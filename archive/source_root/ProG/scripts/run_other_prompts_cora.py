#!/usr/bin/env python
"""Run the Cora "other prompts" few-shot node-classification sweep in parallel.

Covers every NodeTask prompt strategy except GPF / GPF-plus / Gprompt (and the
already-done DHBP, the graph-only DAGPrompT, and MultiGprompt which has no
checkpoint), for shots {1,3,5} x pretrains {None, DGI, GraphMAE, GraphCL,
SimGRACE, Edgepred_GPPT, Edgepred_Gprompt}, num_iter=1, epochs=200, seed=42.

Each cell runs ``scripts/run_prog_cell.py`` in its own subprocess (fresh state,
no Excel-write race) and writes a JSON result file. Results are accumulated
into ``results/other_prompts_cora_cells/*.json`` as cells finish (resumable),
and finally exported to ``results/other_prompts_cora_long_v4.csv`` in the same
schema as ``results/pretrain_fewshot_long.csv``.

Usage:
    python scripts/run_other_prompts_cora.py [--workers N] [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO_ROOT / "Experiment" / "pre_trained_model" / "Cora"
OUT_CSV = REPO_ROOT / "results" / "other_prompts_cora_long_v4.csv"
CELL_DIR = REPO_ROOT / "results" / "other_prompts_cora_cells_v4"
LOG_DIR = REPO_ROOT / "scripts" / "baseline_logs" / "other_prompts_cora_v4"

PROMPTS = [
    "All-in-one",
    "GPPT",
    "Prodigy",
    "GraphPrompter",
    "EdgePrompt",
    "EdgePromptplus",
    "UniPrompt",
    "SelfPro",
    "ProNoG",
    "PSP",
]
PRETRAINS = [
    "None",
    "DGI",
    "GraphMAE",
    "GraphCL",
    "SimGRACE",
    "Edgepred_GPPT",
    "Edgepred_Gprompt",
]
SHOTS = [1, 3, 5]


def ckpt_path(pretrain: str) -> str:
    if pretrain == "None":
        return "None"
    return str(CKPT_ROOT / f"{pretrain}.GCN.128hidden_dim.pth")


def cell_key(shot: int, prompt: str, pretrain: str) -> str:
    return f"{shot}_{prompt}_{pretrain}"


def build_cells() -> list[tuple[int, str, str]]:
    return [(s, p, pt) for s in SHOTS for p in PROMPTS for pt in PRETRAINS]


def run_cell(shot: int, prompt: str, pretrain: str, gpu: int, out_json: Path) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_prog_cell.py"),
        str(shot),
        prompt,
        pretrain,
        ckpt_path(pretrain),
        "0",  # device index within the (single visible) GPU
        str(out_json),
    ]
    log_path = LOG_DIR / f"{cell_key(shot, prompt, pretrain)}.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO_ROOT
        )
    return proc.returncode


def export_csv() -> int:
    import pandas as pd

    rows = []
    for shot, prompt, pretrain in build_cells():
        p = CELL_DIR / f"{cell_key(shot, prompt, pretrain)}.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            row = json.load(f)
            if row.get("status") == "complete":
                rows.append(row)

    if not rows:
        print("no results to export", flush=True)
        return 0
    df = pd.DataFrame(rows)
    order = {pt: i for i, pt in enumerate(PRETRAINS)}
    df["_pt"] = df["pretrain"].map(order)
    df = df.sort_values(["shot", "prompt", "_pt"]).drop(columns="_pt")
    cols = [
        "dataset", "shot", "prompt", "pretrain",
        "accuracy_mean", "accuracy_std", "f1_mean", "f1_std",
        "auroc_mean", "auroc_std",
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

    CELL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cells = build_cells()
    print(
        f"total cells: {len(cells)} (shots={SHOTS}, prompts={len(PROMPTS)}, "
        f"pretrains={len(PRETRAINS)})",
        flush=True,
    )

    todo = []
    for shot, prompt, pretrain in cells:
        out_json = CELL_DIR / f"{cell_key(shot, prompt, pretrain)}.json"
        todo.append((shot, prompt, pretrain, out_json))
    print(f"already done: {len(cells) - len(todo)}; remaining: {len(todo)}", flush=True)

    if args.dry_run:
        for i, (shot, prompt, pretrain, _) in enumerate(todo):
            print(f"  {shot}shot {pretrain}+{prompt} -> gpu {i % 2}")
        return 0

    if not todo:
        export_csv()
        return 0

    done_count = 0
    fail_count = 0

    def worker(item, gpu):
        shot, prompt, pretrain, out_json = item
        key = cell_key(shot, prompt, pretrain)
        rc = run_cell(shot, prompt, pretrain, gpu, out_json)
        return key, rc

    def device_queue(gpu, items):
        return [worker(item, gpu) for item in items]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(device_queue, gpu, todo[gpu::args.workers])
                   for gpu in range(args.workers)]
        for future in concurrent.futures.as_completed(futures):
            for key, rc in future.result():
                if rc == 0:
                    done_count += 1
                else:
                    fail_count += 1
                print(f"{key} rc={rc}", flush=True)

    export_csv()
    print(f"done_count={done_count} fail_count={fail_count}", flush=True)
    if fail_count:
        print("some cells failed; re-run this script to retry missing cells", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
