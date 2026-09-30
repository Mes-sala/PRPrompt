#!/usr/bin/env python
"""Export ProG few-shot node-classification results (Cora / CiteSeer) to CSV.

Reads the populated Excel matrices produced by ``bench.py`` /
``scripts/bench_paper_grid.sh`` at

    Experiment/ExcelResults/Node/{1,3,5}shot/{Cora,CiteSeer}/GCN_total_results.xlsx

and writes two flat CSVs under ``results/``:

1. ``pretrain_fewshot_long.csv``      -- one row per (dataset, shot, prompt,
   pretrain) with Accuracy / F1 / AUROC mean±std split into numeric columns.
2. ``pretrain_vs_none_delta.csv``     -- the key ablation: for every
   (dataset, shot, prompt), the Accuracy of each pretrained backbone, the
   ``None`` (from-scratch) baseline, and the delta (pretrain - None).

Usage:
    python scripts/export_pretrain_fewshot_csv.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCEL_DIR = REPO_ROOT / "Experiment" / "ExcelResults" / "Node"
OUT_DIR = REPO_ROOT / "results"

DATASETS = ["Cora", "CiteSeer"]
SHOTS = [1, 3, 5]
METRIC_ROWS = {
    "accuracy": "Final Accuracy",
    "f1": "Final F1",
    "auroc": "Final AUROC",
}
# Column order: None first so the from-scratch baseline reads as the reference.
PRETRAIN_ORDER = [
    "None",
    "DGI",
    "GraphMAE",
    "GraphCL",
    "SimGRACE",
    "Edgepred_GPPT",
    "Edgepred_Gprompt",
]


def _parse_mean_std(value: object) -> tuple[float | None, float | None]:
    m = re.match(r"([-+0-9.eE]+)±([-+0-9.eE]+)", str(value).strip())
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def load_long() -> pd.DataFrame:
    rows: list[dict] = []
    for shot in SHOTS:
        for ds in DATASETS:
            path = EXCEL_DIR / f"{shot}shot" / ds / "GCN_total_results.xlsx"
            df = pd.read_excel(path, index_col=0)
            for col in df.columns:
                pretrain, prompt = col.split("+", 1)
                rec: dict = {"dataset": ds, "shot": shot, "prompt": prompt, "pretrain": pretrain}
                for out_name, row_name in METRIC_ROWS.items():
                    mean, std = _parse_mean_std(df.loc[row_name, col])
                    rec[f"{out_name}_mean"] = mean
                    rec[f"{out_name}_std"] = std
                rows.append(rec)
    long = pd.DataFrame(rows)
    long["shot"] = long["shot"].astype(int)
    long["pretrain"] = pd.Categorical(long["pretrain"], categories=PRETRAIN_ORDER, ordered=True)
    long = long.sort_values(["dataset", "shot", "prompt", "pretrain"]).reset_index(drop=True)
    return long


def build_delta(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot accuracy so each pretrain sits beside the None baseline + delta."""
    acc = long.pivot_table(
        index=["dataset", "shot", "prompt"],
        columns="pretrain",
        values="accuracy_mean",
    ).reset_index()

    out_rows: list[dict] = []
    base = "None"
    for _, r in acc.iterrows():
        none_acc = r[base]
        for pt in PRETRAIN_ORDER:
            if pt == base:
                continue
            out_rows.append(
                {
                    "dataset": r["dataset"],
                    "shot": r["shot"],
                    "prompt": r["prompt"],
                    "pretrain": pt,
                    "accuracy_none": none_acc,
                    "accuracy_pretrain": r[pt],
                    "accuracy_delta": (r[pt] - none_acc) if pd.notna(r[pt]) and pd.notna(none_acc) else None,
                }
            )
    delta = pd.DataFrame(out_rows)
    delta["shot"] = delta["shot"].astype(int)
    return delta


def main() -> None:
    long = load_long()
    delta = build_delta(long)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_path = OUT_DIR / "pretrain_fewshot_long.csv"
    delta_path = OUT_DIR / "pretrain_vs_none_delta.csv"
    long.to_csv(long_path, index=False)
    delta.to_csv(delta_path, index=False, float_format="%.4f")

    print(f"wrote {long_path} ({len(long)} rows)")
    print(f"wrote {delta_path} ({len(delta)} rows)")


if __name__ == "__main__":
    main()
