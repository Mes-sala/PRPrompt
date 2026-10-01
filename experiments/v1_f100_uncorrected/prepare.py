"""Seal four original checkpoints, paired old metrics."""
import argparse
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import sha_file, save_json
from experiments.v1_f100.run import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--audit-reference", type=Path, required=True)
    args = parser.parse_args()
    refroot = args.workspace/"references/uncorrected_main"
    base = refroot/"experiments/outputs/paper/99a143ae8c31dd31"
    old = json.loads((base/"run_plan.json").read_text())
    assert not old["smoke"] and json.loads((base/"coverage.json").read_text())["present"] == 4680
    assert old["config"]["split_seeds"] == list(range(30093000, 30093020)) and old["config"]["train_seeds"] == [0, 1, 2]
    jobs = []
    for original in old["jobs"]:
        if original["family"] not in ["Edgepred_Gprompt", "Edgepred_GPPT"]:
            continue
        job = dict(original)
        source = args.checkpoint_root/(job["id"]+".pth")
        assert sha_file(source) == job["checkpoint_sha256"]
        target = args.workspace/"checkpoints"/source.name
        if target.exists():
            assert sha_file(target) == job["checkpoint_sha256"]
        else:
            shutil.copy2(source, target)
        model = TwoLayerGCN(job["input_dim"], 128, 128)
        model.load_state_dict(torch.load(target, map_location="cpu", weights_only=True), strict=True)
        job.update(checkpoint=str(target), pretrain_seed="unknown_original", original_checkpoint_path=original["checkpoint"])
        jobs.append(job)
    assert len(jobs) == 4
    references = {"main": {"root": str(refroot), "plan_file": str(base/"run_plan.json"),
                  "plan_file_sha256": sha_file(base/"run_plan.json"),
                  "manifest_file": str(refroot/"experiments/local/result_files_sha256.json"),
                  "manifest_file_sha256": sha_file(refroot/"experiments/local/result_files_sha256.json")}}
    source = {str(p.relative_to(ROOT)): sha_file(p)
              for directory in [ROOT/"prprompt", ROOT/"experiments/v1_f100", ROOT/"experiments/v1_f100_uncorrected", ROOT/"tools"]
              for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ['.py', '.md']}
    plan = {"schema": "v1-F100-original-uncorrected-Edgepred-v1", "base": str(base), "references": references,
            "jobs": jobs,
            "audit_reference_source": str(args.audit_reference), "data_root": "/home/shq/ProG/data",
            "split_seeds": old["config"]["split_seeds"], "train_seeds": [0, 1, 2], "expected_units": 240,
            "shots": 5, "steps": 100, "method": "v1_F100", "split_shards": 3, "source": source,
            "seed_for_bootstrap": 20261001, "primary_reference": "PRPrompt", "pretraining_seed_known": False,
            "query_labels_used_for_fit_or_selection": False, "audit_policy": "predeclared serialized-state probability and raw-forward checks; full-chain numerical drift separately reported"}
    plan["plan_sha256"] = digest(plan)
    output = args.workspace/"plan.json"
    if output.exists():
        assert json.loads(output.read_text()) == plan
    else:
        save_json(output, plan)
    print(json.dumps({"plan": str(output), "sha256": plan["plan_sha256"], "expected": 240}))


if __name__ == "__main__":
    main()
