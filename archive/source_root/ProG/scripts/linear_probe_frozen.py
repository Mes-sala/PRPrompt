#!/usr/bin/env python
"""Frozen-encoder linear probing over ProG's few-shot node splits.

Goal
----
Judge the *representation quality* of each pretrained GCN checkpoint on
Cora / CiteSeer by **freezing the pretrained encoder and training only a
linear classification head** on the few-shot labeled nodes, then evaluating
on the held-out test set. The ``None`` pretrain (random-init, also frozen)
is the from-scratch reference.

This is distinct from the shipped ProG baselines:
  * ``prompt_type='None'``        -> trains gnn + head end-to-end (full finetune)
  * ``prompt_type='GPF'`` etc.    -> frozen gnn + trained prompt + head
  * **this script**               -> frozen gnn + trained head (linear probe)

Protocol (kept identical across every pretrain for a fair ranking)
------------------------------------------------------------------
* Backbone GCN, 2 layers, hid_dim=128 (same arch the checkpoints were saved from).
* Encoder frozen (``requires_grad=False``, ``eval()``); node embeddings are
  computed **once per fold** and reused (true linear probe).
* Head = ``nn.Linear(hid_dim, num_classes)``, Adam(lr, wd), CrossEntropyLoss.
* 5 folds (``task_num=5``), the exact same splits ``bench.py`` uses
  (``Experiment/sample_data/Node/<ds>/<shot>_shot/{1..5}``).
* 200 epochs with early stopping (patience=20 on train loss), matching the
  benchmark budget; the final head (not a best-loss snapshot) is evaluated.

Outputs (under ``results/``):
  * ``linear_probe_frozen_long.csv``        -- one row per (ds, shot, pretrain)
  * ``linear_probe_frozen_vs_none_delta.csv`` -- each pretrain vs ``None`` delta

Usage:
  python scripts/linear_probe_frozen.py --datasets "Cora CiteSeer" --shots "1 3 5" --device 0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from prompt_graph.data import load4node
from prompt_graph.evaluation import GNNNodeEva
from prompt_graph.model import build_gnn
from prompt_graph.utils import get_logger, resolve_device, sample_dir, seed_everything

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ORDER = [
    "None",
    "DGI",
    "GraphMAE",
    "GraphCL",
    "SimGRACE",
    "Edgepred_GPPT",
    "Edgepred_Gprompt",
]


def ckpt_path(dataset: str, pretrain: str, gnn_type: str) -> Path | None:
    if pretrain == "None":
        return None
    return (
        REPO_ROOT
        / "Experiment"
        / "pre_trained_model"
        / dataset
        / f"{pretrain}.{gnn_type}.128hidden_dim.pth"
    )


def probe_fold(
    data,
    input_dim: int,
    output_dim: int,
    fold_dir: Path,
    ckpt: Path | None,
    device: torch.device,
    gnn_type: str,
    hid_dim: int,
    num_layer: int,
    lr: float,
    wd: float,
    epochs: int,
    patience: int,
) -> tuple[float, float, float, float]:
    # 1) build encoder, load pretrained weights, freeze
    gnn = build_gnn(gnn_type, input_dim, hid_dim, num_layer).to(device)
    if ckpt is not None:
        gnn.load_state_dict(torch.load(ckpt, map_location="cpu"))
        gnn.to(device)
    for p in gnn.parameters():
        p.requires_grad = False
    gnn.eval()

    # 2) frozen embeddings, computed once
    with torch.no_grad():
        embeds = gnn(data.x, data.edge_index, batch=None)  # [N, hid_dim]

    # 3) linear head only
    head = nn.Linear(hid_dim, output_dim).to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    idx_train = torch.load(os.path.join(fold_dir, "train_idx.pt")).type(torch.long).to(device)
    idx_test = torch.load(os.path.join(fold_dir, "test_idx.pt")).type(torch.long).to(device)
    y = data.y.to(device)

    best = float("inf")
    cnt_wait = 0
    for _ in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        out = head(embeds)
        loss = criterion(out[idx_train], y[idx_train])
        loss.backward()
        optimizer.step()
        if loss.item() < best:
            best = loss.item()
            cnt_wait = 0
        else:
            cnt_wait += 1
            if cnt_wait == patience:
                break

    # 4) evaluate with the same torchmetrics code as the benchmark
    acc, f1, roc, prc = GNNNodeEva(data, idx_test, gnn, head, output_dim, device)
    return float(acc), float(f1), float(roc), float(prc)


def run_dataset(
    dataset: str,
    shots: list[int],
    pretrains: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    data, input_dim, output_dim = load4node(dataset)
    data = data.to(device)
    rows: list[dict] = []
    for shot in shots:
        base = sample_dir("Node", shot, dataset)
        fold_dirs = [base / str(i) for i in range(1, args.task_num + 1)]
        for pretrain in pretrains:
            ckpt = ckpt_path(dataset, pretrain, args.gnn_type)
            if ckpt is not None and not ckpt.exists():
                logger.warning("missing checkpoint %s, skipping", ckpt)
                continue
            accs, f1s, rocs, prcs = [], [], [], []
            for fold_dir in fold_dirs:
                acc, f1, roc, prc = probe_fold(
                    data,
                    input_dim,
                    output_dim,
                    fold_dir,
                    ckpt,
                    device,
                    args.gnn_type,
                    args.hid_dim,
                    args.num_layer,
                    args.lr,
                    args.wd,
                    args.epochs,
                    args.patience,
                )
                accs.append(acc)
                f1s.append(f1)
                rocs.append(roc)
                prcs.append(prc)
            rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "pretrain": pretrain,
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "auroc_mean": float(np.mean(rocs)),
                    "auroc_std": float(np.std(rocs)),
                    "auprc_mean": float(np.mean(prcs)),
                    "auprc_std": float(np.std(prcs)),
                }
            )
            logger.info(
                "%s %d-shot %s: acc %.4f±%.4f",
                dataset, shot, pretrain, rows[-1]["accuracy_mean"], rows[-1]["accuracy_std"],
            )
    return rows


def build_delta(long: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict] = []
    base = "None"
    for (ds, shot), grp in long.groupby(["dataset", "shot"]):
        none_acc = grp.loc[grp["pretrain"] == base, "accuracy_mean"].item()
        for _, r in grp.iterrows():
            if r["pretrain"] == base:
                continue
            out_rows.append(
                {
                    "dataset": ds,
                    "shot": shot,
                    "pretrain": r["pretrain"],
                    "accuracy_none": none_acc,
                    "accuracy_pretrain": r["accuracy_mean"],
                    "accuracy_delta": r["accuracy_mean"] - none_acc,
                }
            )
    delta = pd.DataFrame(out_rows)
    delta["pretrain"] = pd.Categorical(delta["pretrain"], categories=PRETRAIN_ORDER[1:], ordered=True)
    return delta.sort_values(["dataset", "shot", "pretrain"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--pretrains", nargs="+", default=PRETRAIN_ORDER)
    ap.add_argument("--gnn_type", default="GCN")
    ap.add_argument("--hid_dim", type=int, default=128)
    ap.add_argument("--num_layer", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--task_num", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)

    rows: list[dict] = []
    for ds in args.datasets:
        rows.extend(run_dataset(ds, args.shots, args.pretrains, device, args))

    long = pd.DataFrame(rows)
    long["pretrain"] = pd.Categorical(long["pretrain"], categories=PRETRAIN_ORDER, ordered=True)
    long = long.sort_values(["dataset", "shot", "pretrain"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "linear_probe_frozen_long.csv"
    delta_path = out_dir / "linear_probe_frozen_vs_none_delta.csv"
    long.to_csv(long_path, index=False, float_format="%.4f")
    build_delta(long).to_csv(delta_path, index=False, float_format="%.4f")
    print(f"wrote {long_path} ({len(long)} rows)")
    print(f"wrote {delta_path}")


if __name__ == "__main__":
    main()
