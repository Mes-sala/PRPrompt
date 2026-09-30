"""Resolve existing checkpoints, verify bytes and architecture, never silently substitute."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import sha_file, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-plan", default="runtime/base/plan.json")
    ap.add_argument("--archive-root", type=Path, required=True, help="Directory corresponding to historical /root")
    ap.add_argument("--runtime-root", type=Path, help="Existing 5090 softening directory, if available")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--families", nargs="+")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = json.loads(Path(args.source_plan).read_text())
    jobs, unavailable, seen = [], [], set()
    for entry in source["jobs"]:
        cp = entry.get("checkpoint")
        if not cp or entry.get("direct"):
            continue
        if args.datasets and entry["dataset"] not in args.datasets:
            continue
        if args.families and cp["family"] not in args.families:
            continue
        identity = (entry["dataset"], cp["family"], cp["sha256"])
        if identity in seen:
            continue
        seen.add(identity)
        candidates = []
        if args.runtime_root:
            candidates.append(args.runtime_root / "checkpoints" / (entry["id"] + ".pth"))
        for old in [cp.get("path", "")] + cp.get("paths", []):
            if old.startswith("/root/"):
                candidates.append(args.archive_root / old.removeprefix("/root/"))
        path = next((p for p in candidates if p.is_file() and sha_file(p) == cp["sha256"]), None)
        if path is None:
            unavailable.append({"id": entry["id"], "reason": "checkpoint_missing_or_hash_mismatch"})
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        try:
            h, d = state["conv_layers.0.lin.weight"].shape
            out, h2 = state["conv_layers.1.lin.weight"].shape
            if h != h2:
                raise ValueError("Hidden dimensions differ")
            TwoLayerGCN(d, h, out).load_state_dict(state, strict=True)
        except (KeyError, RuntimeError, ValueError) as e:
            unavailable.append({"id": entry["id"], "reason": f"incompatible_two_layer_gcn: {e}"})
            continue
        provenance = cp.get("provenance", {})
        if provenance.get("labels_used") is True:
            unavailable.append({"id": entry["id"], "reason": "supervised_checkpoint_not_in_unsupervised_track"})
            continue
        jobs.append({"id": entry["id"], "dataset": entry["dataset"], "family": cp["family"],
                     "checkpoint": str(path.resolve()), "checkpoint_sha256": cp["sha256"],
                     "input_dim": d, "hidden_dim": h, "output_dim": out,
                     "pretraining_provenance": provenance})
    save_json(args.output, {"schema": "prprompt-v2-resolved-checkpoints", "source_plan_sha256": sha_file(args.source_plan),
                           "data_root": str(args.data_root.resolve()), "jobs": jobs, "unavailable": unavailable})
    print(json.dumps({"ready": len(jobs), "unavailable": len(unavailable), "output": str(args.output)}))
    if unavailable or not jobs:
        raise SystemExit("Manifest incomplete: inspect unavailable; do not report missing methods as results")


if __name__ == "__main__":
    main()
