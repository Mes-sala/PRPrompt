#!/usr/bin/env python
"""Run a single ProG NodeTask bench cell and write the result as JSON to a file.

This is the per-cell worker used by ``scripts/run_other_prompts_cora.py``.
Unlike ``bench.py`` (whose ``__main__`` block writes into the shared Excel
matrix), this script calls ``do_config_bench`` directly and returns the result
in-memory, so many cells can run in parallel without racing on the same Excel
file.

The result is written to ``--out`` (atomically) rather than stdout, because
the tasker/prompt modules still emit raw ``print()`` output to stdout and would
pollute a JSON stream.

Usage:
    python scripts/run_prog_cell.py <shot> <prompt> <pretrain> <ckpt|None> <device> <out_json>
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# ``bench`` lives at the repo root (not installed as a package), so put the
# repo root on sys.path before importing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def _write_json_atomic(path: str, payload: dict) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".cell-", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    if len(sys.argv) != 7:
        print(
            "usage: run_prog_cell.py <shot> <prompt> <pretrain> <ckpt|None> <device> <out_json>",
            file=sys.stderr,
        )
        return 2

    shot = int(sys.argv[1])
    prompt = sys.argv[2]
    pretrain = sys.argv[3]
    ckpt = sys.argv[4]
    device = sys.argv[5]
    out_path = sys.argv[6]

    # Compatibility CLI, routed to the explicit v4 engine. Historical JSONs
    # are never reused by the versioned matrix scheduler.
    from run_protocol_cell import run
    config = dict(dataset="Cora", prompt=prompt, checkpoint=ckpt, gnn="GCN",
                  hidden_dim=128, layers=2, shot=shot, folds=5,
                  support_passes=200, learning_rate=1e-3, weight_decay=1e-5,
                  split_seed=1729, train_seed=42, inference_seed=1000042,
                  device=device, output_root=str(Path(out_path).parent / "artifacts"),
                  split_root=str(Path(__file__).resolve().parents[1] / "Experiment/sample_data_v4"))
    if prompt == "GPPT":
        config.update(learning_rate=2e-3, weight_decay=5e-4)
    if prompt == "All-in-one":
        config.update(answer_passes=50, prompt_passes=50, prompt_lr=1e-6)
    try:
        result = run(config)
    except NotImplementedError as exc:
        _write_json_atomic(out_path, dict(status="skipped", prompt=prompt,
                                         pretrain=pretrain, reason=str(exc)))
        return 0
    except Exception as exc:
        _write_json_atomic(out_path, dict(status="failed", prompt=prompt, pretrain=pretrain, reason=str(exc)))
        raise
    values = result["aggregate"]
    rec = dict(dataset="Cora", shot=shot, prompt=prompt, pretrain=pretrain,
               status="complete", protocol=result["protocol"], run_id=result["run_id"],
               accuracy_mean=values[1], accuracy_std=values[2],
               f1_mean=values[3], f1_std=values[4], auroc_mean=values[5], auroc_std=values[6],
               artifact_directory=result["output_directory"])
    _write_json_atomic(out_path, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
