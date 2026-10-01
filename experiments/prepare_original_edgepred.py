"""Resolve the pinned original Edgepred weights through the standard ProG paths."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import save_json, sha_file


def prepare(prog_root, datasets, output):
    catalog_path = ROOT / "experiments/configs/original_edgepred_checkpoints.json"
    catalog = json.loads(catalog_path.read_text())
    known = {entry["dataset"] for entry in catalog["checkpoints"]}
    unknown = set(datasets or []) - known
    if unknown:
        raise ValueError(f"No pinned original checkpoints for: {sorted(unknown)}")
    jobs = []
    for entry in catalog["checkpoints"]:
        if datasets and entry["dataset"] not in datasets:
            continue
        path = prog_root / "Experiment/pre_trained_model" / entry["dataset"] / (
            entry["family"] + ".GCN.128hidden_dim.pth")
        if not path.is_file() or sha_file(path) != entry["checkpoint_sha256"]:
            raise ValueError(f"Original checkpoint missing or changed: {path}")
        model = TwoLayerGCN(entry["input_dim"], entry["hidden_dim"], entry["output_dim"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        provenance = {}
        for suffix in [".provenance.json", ".json"]:
            sidecar = Path(str(path) + suffix)
            if not sidecar.exists():
                continue
            metadata = json.loads(sidecar.read_text())
            if metadata.get("checkpoint_sha256", entry["checkpoint_sha256"]) != entry["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint provenance mismatch: {sidecar}")
            if (metadata.get("labels_used") is True or metadata.get("node_label_values_used") is True
                   ):
                raise ValueError(f"Not an original unsupervised checkpoint: {sidecar}")
            provenance.update(metadata)
        job = dict(entry)
        job.update(id=entry["dataset"] + "__" + entry["family"] + "__" + entry["checkpoint_sha256"][:12],
                   checkpoint=str(path.resolve()), pretrain_seed="unknown_original",
                   pretraining_provenance=provenance)
        jobs.append(job)
    if not jobs:
        raise ValueError("No original checkpoints selected")
    manifest = dict(schema="prprompt-v2-resolved-checkpoints", checkpoint_version="original_uncorrected",
                    catalog_sha256=sha_file(catalog_path), data_root=str((prog_root / "data").resolve()),
                    jobs=jobs, unavailable=[])
    save_json(output, manifest)
    print(json.dumps(dict(ready=len(jobs), checkpoint_version="original_uncorrected", output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prog-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.prog_root.resolve(), args.datasets, args.output)
