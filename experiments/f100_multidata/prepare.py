"""Snapshot inputs and freeze the multi-dataset experiment before fitting."""
import argparse
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import load_dataset, save_json, sha_file
from prprompt.train import tensor_hash
from experiments.v1_f100_uncorrected.run import digest

METHODS = ["FrozenLinear", "FrozenPrototype", "FrozenPrototypeCentered", "FullFineTune",
           "GPF", "GPFPlus", "GraphPromptDiag", "PRPrompt", "F100"]


def prepare(workspace):
    assert not (workspace/"plan.json").exists(), "Do not replace a frozen plan"
    inventory = json.loads((workspace/"preflight_inventory.json").read_text())
    config = json.loads((ROOT/"experiments/configs/main.json").read_text())
    jobs, datasets = [], {}
    for item in inventory:
        name = item["dataset"]
        x, edges, labels, pool, query, classes = load_dataset(name, workspace/"data")
        assert torch.bincount(labels[pool], minlength=classes).min() >= 5
        assert len(labels[query].unique()) == classes
        datasets[name] = dict(nodes=len(x), input_dim=x.shape[1], edges=edges.shape[1], classes=classes,
            graph_sha256=digest({"x": tensor_hash(x), "edges": tensor_hash(edges)}),
            pool_sha256=tensor_hash(pool), query_sha256=tensor_hash(query), labels_sha256=tensor_hash(labels),
            split_policy="fixed_stratified_20_per_class_seed20260930" if name in ["Photo", "Computers"] else "official_masks_fold0",
            pool_per_class=torch.bincount(labels[pool], minlength=classes).tolist(),
            query_per_class=torch.bincount(labels[query], minlength=classes).tolist())
        snapshot = workspace/"dataset_snapshots"/(name+".pt")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(x=x, edges=edges, labels=labels, pool=pool, query=query), snapshot)
        datasets[name].update(snapshot=str(snapshot), snapshot_sha256=sha_file(snapshot))
        for checkpoint in item["checkpoints"]:
            family = checkpoint["family"]
            checkpoint_id = name+"_"+family+"_"+checkpoint["sha256"][:12]
            target = workspace/"checkpoints"/(checkpoint_id+".pth")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint["path"], target)
            assert sha_file(target) == checkpoint["sha256"]
            model = TwoLayerGCN(x.shape[1], 128, 128)
            model.load_state_dict(torch.load(target, map_location="cpu", weights_only=True), strict=True)
            for sidecar, value in checkpoint["sidecars"].items():
                assert value["checkpoint_sha256"] == checkpoint["sha256"]
                save_json(target.with_suffix(".pth.provenance.json"), value)
            jobs.append(dict(id=checkpoint_id, dataset=name, family=family, checkpoint=str(target),
                checkpoint_sha256=checkpoint["sha256"], original_checkpoint_path=checkpoint["path"],
                pretrain_seed="unknown_original", input_dim=x.shape[1], hidden_dim=128, output_dim=128,
                provenance_status="ProG-adapted_not_fully_verified" if checkpoint["sidecars"] else "legacy_unverified"))
    assert len(jobs) == 10
    source = {}
    for dirname in ["prprompt", "experiments/v1_f100", "experiments/f100_multidata"]:
        for path in sorted((ROOT/dirname).rglob("*")):
            if path.is_file() and path.suffix in [".py", ".md"]:
                source[str(path.relative_to(ROOT))] = sha_file(path)
    for relative in ["experiments/v1_f100_uncorrected/run.py", "tools/metric_utils.py",
                     "experiments/configs/main.json", "tools/audit_f100_multidata_delivery.py"]:
        source[relative] = sha_file(ROOT/relative)
    plan = dict(schema="f100-original-edgepred-multidata-v1", jobs=jobs, datasets=datasets,
        source=source, source_root=str(ROOT), data_root=str(workspace/"data"), methods=METHODS,
        split_seeds=config["split_seeds"], train_seeds=config["train_seeds"], split_shards=3,
        config=config, shots=5, f100_steps=100, bootstrap_seed=20261001,
        expected_units=len(jobs)*len(METHODS)*20*3,
        audit_rules=dict(serialized_probability_atol=2e-6, forward_atol=2e-6, forward_rtol=1e-6,
            gate_atol=2e-6, metric_atol=1e-12, strict_full_chain_atol=2e-6,
            strict_full_chain_is_diagnostic=True),
        environment=dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda),
        boundary="One original checkpoint per dataset/family; unknown pretraining seed. Full graph x/A visible, support labels only; no new-class split, no M2/M3/KD or attribute baselines.")
    plan["plan_sha256"] = digest(plan)
    save_json(workspace/"plan.json", plan)
    print(json.dumps({"expected": plan["expected_units"], "jobs": len(jobs), "plan_sha256": plan["plan_sha256"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    prepare(parser.parse_args().workspace)
