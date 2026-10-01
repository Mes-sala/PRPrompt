"""Execute the preregistered comparison; query labels enter only after sealing."""
import argparse
import hashlib
import fcntl
import itertools
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from prprompt.model import TwoLayerGCN
from prprompt.train import fit_predict, tensor_hash, MAIN_METHODS, DIAGNOSTIC_METHODS
from prprompt.protocol import (load_dataset, sample_support, select_config, sha_file, save_json,
                               seal_predictions, evaluate_sealed)


def code_manifest():
    paths = sorted((ROOT / "prprompt").glob("*.py")) + sorted((ROOT / "experiments").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha_file(p) for p in paths}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run(args):
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("unavailable"):
        raise ValueError("Resolve unavailable checkpoints before running")
    jobs = [j for j in manifest["jobs"] if not args.checkpoint_id or j["id"] in args.checkpoint_id]
    if not jobs:
        raise ValueError("No selected jobs")
    if args.methods:
        config["methods"] = args.methods
    if set(config["methods"]) - set(MAIN_METHODS + DIAGNOSTIC_METHODS):
        raise ValueError("Unknown method in configuration")
    for key in ["methods", "shots", "split_seeds", "train_seeds"]:
        if not config[key] or len(config[key]) != len(set(config[key])):
            raise ValueError(f"Empty or duplicate {key}")
    if args.smoke:
        config["purpose"] = "smoke_not_paper_evidence"
        config["shots"] = [1, 5]
        config["split_seeds"] = config["split_seeds"][:1]
        config["train_seeds"] = config["train_seeds"][:1]
        config["fit"]["steps"] = 2
        config["selection"] = {"learning_rates": [config["fit"]["lr"]], "weight_decays": [config["fit"]["weight_decay"]]}
    planned = {"config": config, "jobs": jobs, "data_root": manifest["data_root"], "source": code_manifest(),
               "smoke": args.smoke, "manifest_sha256": sha_file(args.manifest)}
    planned["expected_runs"] = len(jobs) * len(config["methods"]) * len(config["shots"]) * len(config["split_seeds"]) * len(config["train_seeds"])
    plan_hash = digest(planned)
    output = args.output / ("smoke" if args.smoke else "paper") / plan_hash[:16]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "plan_write.lock").open("a") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        if (output / "run_plan.json").exists():
            existing = json.loads((output / "run_plan.json").read_text())
            if existing != dict(planned, plan_sha256=plan_hash):
                raise ValueError("Existing execution plan differs")
        else:
            save_json(output / "run_plan.json", dict(planned, plan_sha256=plan_hash))
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
    workers, worker = args.workers, args.worker_index
    if not 0 <= worker < workers:
        raise ValueError("Worker index must be within worker count")
    print(json.dumps({"output": str(output), "expected_runs": planned["expected_runs"], "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        return
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    completed = 0
    unit_index = 0
    assigned = 0
    for job in jobs:
        if sha_file(job["checkpoint"]) != job["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {job['id']}")
        x, edges, labels, pool, query, classes = load_dataset(job["dataset"], manifest["data_root"])
        if x.shape[1] != job["input_dim"]:
            raise ValueError("Data/checkpoint feature dimension mismatch")
        graph_hash = digest({"x": tensor_hash(x), "edges": tensor_hash(edges)})
        split_policy_hash = digest({"pool": tensor_hash(pool), "query": tensor_hash(query)})
        backbone = TwoLayerGCN(job["input_dim"], job["hidden_dim"], job["output_dim"])
        backbone.load_state_dict(torch.load(job["checkpoint"], map_location="cpu", weights_only=True), strict=True)
        x, edges = x.to(device), edges.to(device)
        for shots, split_seed, train_seed, method in itertools.product(config["shots"], config["split_seeds"], config["train_seeds"], config["methods"]):
            selected = unit_index % workers == worker
            unit_index += 1
            if not selected:
                continue
            assigned += 1
            support = sample_support(pool, labels[pool], classes, shots, split_seed)
            if torch.isin(support, query).any():
                raise ValueError("Support/query overlap")
            directory = output / job["id"] / str(shots) / str(split_seed) / str(train_seed) / method
            directory.mkdir(parents=True, exist_ok=True)
            metadata = {"plan_sha256": plan_hash, "dataset": job["dataset"], "checkpoint_id": job["id"],
                        "family": job["family"], "checkpoint_sha256": job["checkpoint_sha256"],
                        "graph_sha256": graph_hash, "split_policy_sha256": split_policy_hash,
                        "split_seed": split_seed, "train_seed": train_seed, "shots": shots, "method": method,
                        "support_labels_sha256": tensor_hash(labels[support]), "smoke": args.smoke}
            signature = digest(metadata)
            report_path = directory / "metrics.json"
            if report_path.exists():
                old = json.loads(report_path.read_text())
                if old["signature"] != signature or sha_file(directory / "predictions.npz") != old["predictions_sha256"]:
                    raise ValueError(f"Resume integrity mismatch: {directory}")
                if sha_file(directory / "state.pt") != old["state_sha256"] or sha_file(directory / "selection.json") != old["selection_sha256"]:
                    raise ValueError(f"Resume state/selection changed: {directory}")
                if old["evaluation_labels_sha256"] != tensor_hash(labels[query]):
                    raise ValueError(f"Evaluation labels changed: {directory}")
                completed += 1
                continue
            lock = directory / "running.lock"
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            try:
                started = time.perf_counter()
                s, sy = support.to(device), labels[support].to(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                chosen, selection = select_config(method, backbone, x, edges, s, sy, classes, config, train_seed)
                save_json(directory / "selection.json", selection)
                scores, state, audit = fit_predict(method, backbone, x, edges, s, sy, classes, chosen, train_seed)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                torch.save(state, directory / "state.pt")
                seal = seal_predictions(directory, scores, support, query.to(device), dict(metadata, signature=signature))
                # Query label access for scoring occurs only AFTER prediction bytes and seal are saved.
                metrics = evaluate_sealed(directory, labels[query])
                report = dict(metadata, signature=signature, predictions_sha256=seal["predictions_sha256"],
                              state_sha256=sha_file(directory / "state.pt"), selection_sha256=sha_file(directory / "selection.json"),
                              support_sha256=seal["support_sha256"], query_sha256=seal["query_sha256"], metrics=metrics,
                              evaluation_labels_sha256=tensor_hash(labels[query]),
                              selected_config=chosen, audit=audit, total_seconds=time.perf_counter() - started,
                              peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                              environment={"torch": str(torch.__version__), "device": str(device), "cuda": torch.version.cuda})
                save_json(report_path, report)
                completed += 1
                print(json.dumps({"complete": completed, "expected": planned["expected_runs"], "job": job["id"],
                                  "method": method, "shots": shots, "split_seed": split_seed, "train_seed": train_seed}), flush=True)
            finally:
                lock.unlink()
    if completed != assigned:
        raise RuntimeError("Incomplete run")
    save_json(output / f"worker_completion_{worker}.json", {"plan_sha256": plan_hash,
              "worker_index": worker, "workers": workers, "completed": completed,
              "assigned": assigned, "smoke": args.smoke})
    if workers == 1:
        save_json(output / "completion.json", {"plan_sha256": plan_hash, "completed": completed, "smoke": args.smoke})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/configs/main.json")
    ap.add_argument("--output", type=Path, default=ROOT / "experiments/outputs")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--checkpoint-id", nargs="+")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-index", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
