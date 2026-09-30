#!/usr/bin/env python
"""Run the Cora 5-shot full matrix under the v4 protocol.

Each (pretrain, prompt) cell is submitted to ``scripts/run_protocol_cell.py``
with an explicit v4 config file. The v4 engine writes a content-addressed
artifact bundle (``config.json``/``identity.json``/``result.json``/
``predictions.json``) under ``output_root/<run_id>/``; the run_id is derived
from the protocol + config + data + checkpoint + source hashes, so re-running
the same cell reuses the cached artifact (no legacy flat-JSON roundtrip).

This driver only records a lightweight *index* per cell pointing at the v4
artifact directory -- it never emits or consumes the legacy flat-metrics JSON
(``accuracy_mean`` etc.), which is the legacy compatibility path being removed.
Metrics for the CSV are read back from each artifact's ``result.json``.

Pairing rules:
  * prompt == None            -> every pretrain (scratch + pretrained full finetune)
  * prompt == MultiGprompt    -> only pretrain == MultiGprompt (self-contained pair)
  * pretrain == MultiGprompt  -> only prompt == MultiGprompt
  * otherwise                 -> every regular pretrain x every regular prompt

Methods not in the v4 capability table (or unverified without
``allow_unverified``) are recorded as ``skipped`` by the engine, not faked as 0.

Usage:
    python scripts/run_cora_5shot_matrix.py [--workers N] [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO_ROOT / "Experiment" / "pre_trained_model" / "Cora"
OUT_CSV = REPO_ROOT / "results" / "cora_5shot_full_matrix_v4.csv"
CELL_DIR = REPO_ROOT / "results" / "cora_5shot_cells_v4"
ARTIFACT_ROOT = CELL_DIR / "artifacts"
LOG_DIR = REPO_ROOT / "scripts" / "baseline_logs" / "cora_5shot_matrix_v4"
SPLIT_ROOT = REPO_ROOT / "Experiment" / "sample_data_v4"

SHOT = 5

# Task heads enumerated in the matrix (kept as the legacy 24-method list;
# methods outside the v4 capability table are recorded as skipped).
PROMPTS = [
    "None",
    "GPF",
    "GPF-plus",
    "Gprompt",
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
    "Frozen",
    "FrozenNodePrototype",
    "FrozenIdentityPrototype",
    "GraphTOP-Frozen",
    "FrozenGraph",
    "FrozenGraphPrototype",
    "DHBP",
    "SUPT-soft",
    "SUPT-hard",
    "GraphTOP",
]

# Regular pretrains cross-combined with every regular prompt.
PRETRAINS = [
    "None",
    "DGI",
    "GraphMAE",
    "GraphMAE2",
    "S2GAE",
    "Bandana",
    "Edgepred_GPPT",
    "Edgepred_Gprompt",
    "GraphCL",
    "SimGRACE",
]

# metric_order name -> CSV column alias (aggregate = [loss, acc_mean, acc_std,
# f1_mean, f1_std, auroc_mean, auroc_std, auprc_mean, auprc_std]).
METRIC_ALIASES = {
    "accuracy": "accuracy",
    "macro_f1": "f1",
    "macro_auroc_ovr": "auroc",
    "macro_auprc_ovr": "auprc",
}


def ckpt_path(pretrain: str) -> str:
    if pretrain == "None":
        return "None"
    if pretrain == "MultiGprompt":
        return str(CKPT_ROOT / "MultiGprompt.pth")
    return str(CKPT_ROOT / f"{pretrain}.GCN.128hidden_dim.pth")


def cell_key(prompt: str, pretrain: str) -> str:
    return f"{prompt}_{pretrain}"


def build_cells() -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for pretrain in PRETRAINS:
        for prompt in PROMPTS:
            # Include pretrained full-finetune baselines in v4 (None prompt x
            # every pretrain, per the runbook).
            cells.append((prompt, pretrain))
    # MultiGprompt is a self-contained pretrain+strategy pair.
    cells.append(("MultiGprompt", "MultiGprompt"))
    return cells


def build_config(prompt: str, pretrain: str, device: int) -> dict:
    config = dict(
        dataset="Cora",
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
        split_seed=1729,
        train_seed=42,
        inference_seed=1000042,
        device=str(device),
        subgraph_min=100,
        subgraph_max=300,
        output_root=str(ARTIFACT_ROOT),
        split_root=str(SPLIT_ROOT),
    )
    if prompt == "GPPT":
        config.update(learning_rate=2e-3, weight_decay=5e-4)
    if prompt == "All-in-one":
        config.update(answer_passes=50, prompt_passes=50, prompt_lr=1e-6)
    return config


def metrics_from_result(result: dict) -> dict:
    """Map the v4 ``aggregate`` list to named mean/std columns via metric_order."""
    order = result["metric_order"]
    agg = result["aggregate"]
    out: dict = {"loss_mean": agg[0]}
    i = 1
    for name in order:
        out[f"{name}_mean"] = agg[i]
        out[f"{name}_std"] = agg[i + 1]
        i += 2
    return out


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_cell(prompt: str, pretrain: str, device: int) -> int:
    key = cell_key(prompt, pretrain)
    config = build_config(prompt, pretrain, device)
    config_path = CELL_DIR / f"{key}.config.json"
    index_path = CELL_DIR / f"{key}.json"
    write_json(config_path, config)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_protocol_cell.py"),
        "--config",
        str(config_path),
    ]
    log_path = LOG_DIR / f"{key}.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO_ROOT
        )

    # run_protocol_cell.py prints the status JSON as its final stdout line
    # (task modules may print intermediate "Final best ..." lines before it).
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
            shot=SHOT,
            prompt=prompt,
            pretrain=pretrain,
            protocol=result.get("protocol"),
            run_id=result.get("run_id"),
            artifact_directory=str(artifact),
            variant=result.get("variant"),
            input_mode=result.get("input_mode"),
            readout=result.get("readout"),
            head=result.get("head"),
            checkpoint_provenance=result.get("checkpoint_provenance"),
        )
        write_json(index_path, index)
        return 0

    if status_name == "skipped":
        write_json(index_path, dict(
            status="skipped", dataset="Cora", shot=SHOT, prompt=prompt,
            pretrain=pretrain, reason=status.get("reason")))
        return 0

    write_json(index_path, dict(
        status="failed", dataset="Cora", shot=SHOT, prompt=prompt,
        pretrain=pretrain, reason=status.get("reason"),
        exit_code=proc.returncode))
    return 1


def export_csv() -> int:
    import pandas as pd

    rows = []
    for prompt, pretrain in build_cells():
        index_path = CELL_DIR / f"{cell_key(prompt, pretrain)}.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("status") != "complete":
            rows.append(index)
            continue
        result = json.loads(
            (Path(index["artifact_directory"]) / "result.json").read_text(encoding="utf-8")
        )
        m = metrics_from_result(result)
        row = {
            "dataset": index.get("dataset"),
            "shot": index.get("shot"),
            "prompt": index.get("prompt"),
            "pretrain": index.get("pretrain"),
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
        for name, alias in METRIC_ALIASES.items():
            row[f"{alias}_mean"] = m[f"{name}_mean"]
            row[f"{alias}_std"] = m[f"{name}_std"]
        rows.append(row)

    if not rows:
        print("no results to export", flush=True)
        return 0

    df = pd.DataFrame(rows)
    pt_order = {pt: i for i, pt in enumerate(PRETRAINS + ["MultiGprompt"])}
    pr_order = {p: i for i, p in enumerate(PROMPTS + ["MultiGprompt"])}
    df["_pt"] = df["pretrain"].map(pt_order)
    df["_pr"] = df["prompt"].map(pr_order)
    df = df.sort_values(["shot", "_pt", "_pr"]).drop(columns=["_pt", "_pr"])
    cols = [
        "dataset", "shot", "prompt", "pretrain", "status", "protocol", "run_id",
        "artifact_directory", "variant", "input_mode", "readout", "head",
        "checkpoint_provenance",
        "accuracy_mean", "accuracy_std", "f1_mean", "f1_std",
        "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)", flush=True)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("workers must be positive")

    CELL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cells = build_cells()
    print(
        f"total cells: {len(cells)} (shot={SHOT}, prompts={len(PROMPTS)} + MultiGprompt, "
        f"pretrains={len(PRETRAINS)} + MultiGprompt)",
        flush=True,
    )

    todo = []
    for i, (prompt, pretrain) in enumerate(cells):
        index_path = CELL_DIR / f"{cell_key(prompt, pretrain)}.json"
        if index_path.exists():
            try:
                existing = json.loads(index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            # Resubmit only failed cells; reuse complete and skipped outcomes.
            if existing.get("status") in {"complete", "skipped"}:
                continue
        todo.append((prompt, pretrain))
    print(f"already done: {len(cells) - len(todo)}; remaining: {len(todo)}", flush=True)

    if args.dry_run:
        for i, (prompt, pretrain) in enumerate(todo):
            print(f"  {SHOT}shot {pretrain}+{prompt} -> gpu {i % args.workers}")
        return 0

    if not todo:
        return export_csv()

    done_count = 0
    fail_count = 0
    failed_cells: list[str] = []

    # One serial queue per GPU; a cell's device is fixed by its stable index so
    # the resulting run_id is reproducible across resubmissions.
    def worker(item, gpu):
        prompt, pretrain = item
        rc = run_cell(prompt, pretrain, gpu)
        return (prompt, pretrain), rc

    def device_queue(gpu, items):
        return [worker(item, gpu) for item in items]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(device_queue, gpu, todo[gpu::args.workers])
                   for gpu in range(args.workers)]
        for future in concurrent.futures.as_completed(futures):
            for (prompt, pretrain), rc in future.result():
                if rc == 0:
                    done_count += 1
                else:
                    fail_count += 1
                    failed_cells.append(f"{pretrain}+{prompt}")
                print(f"{pretrain}+{prompt} rc={rc}", flush=True)

    n = export_csv()
    print(f"done_count={done_count} fail_count={fail_count}", flush=True)
    if fail_count:
        print("failed cells:", ", ".join(failed_cells), flush=True)
        print("re-run this script to retry failed cells", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
