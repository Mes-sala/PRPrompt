"""Disjoint resumable shards; unchanged F100 and existing support-CV baselines."""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from prprompt.model import TwoLayerGCN, normalized_adjacency
from prprompt.protocol import sample_support, select_config, sha_file, save_json, seal_predictions, evaluate_sealed
from prprompt.train import fit_predict, state_hash, tensor_hash
from experiments.v1_f100.historical import load, fit_f100
from experiments.v1_f100_uncorrected.run import digest


def read_plan(path):
    plan = json.loads(path.read_text())
    identity_key = "document_sha256" if "document_sha256" in plan else "plan_sha256"
    assert digest({k: v for k, v in plan.items() if k != identity_key}) == plan[identity_key]
    for name, expected in plan["source"].items():
        assert sha_file(ROOT/name) == expected, name
    return plan


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = read_plan(args.plan); job = plan["jobs"][args.job_index]
    output = args.plan.parent/("smoke" if args.smoke else "results")/job["id"]
    output.mkdir(parents=True, exist_ok=True)
    lock = output/(f"running_shard_{args.shard}.lock")
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        assert sha_file(job["checkpoint"]) == job["checkpoint_sha256"]
        data_spec = plan["datasets"][job["dataset"]]
        assert sha_file(data_spec["snapshot"]) == data_spec["snapshot_sha256"]
        data = torch.load(data_spec["snapshot"], map_location="cpu", weights_only=True)
        x, edges, labels, pool, query = (data[k] for k in ["x", "edges", "labels", "pool", "query"])
        classes = data_spec["classes"]
        graph_hash = digest({"x": tensor_hash(x), "edges": tensor_hash(edges)})
        assert graph_hash == data_spec["graph_sha256"]
        device = torch.device(args.device); x, edges = x.to(device), edges.to(device)
        model = TwoLayerGCN(job["input_dim"], 128, 128).to(device).eval()
        model.load_state_dict(torch.load(job["checkpoint"], map_location="cpu", weights_only=True), strict=True)
        model.requires_grad_(False); before = state_hash(model)
        adj = normalized_adjacency(edges, len(x), x.dtype); old = load()
        with torch.no_grad():
            u = model.conv_layers[0](x, edges).detach()
            r = torch.sparse.mm(adj, torch.sparse.mm(adj, x)).detach()
            scale = u.std(0).clamp_min(1e-6)
            filters = old["make_filters"](x, edges)
        splits = plan["split_seeds"][:1] if args.smoke else plan["split_seeds"][args.shard::plan["split_shards"]]
        seeds = plan["train_seeds"][:1] if args.smoke else plan["train_seeds"]
        steps = 2 if args.smoke else plan["f100_steps"]
        config = copy.deepcopy(plan["config"])
        if args.smoke:
            config["fit"]["steps"] = 2
            config["selection"] = dict(learning_rates=[.01], weight_decays=[.0001])
        completed = 0
        for split in splits:
            support = sample_support(pool, labels[pool], classes, 5, split)
            sy = labels[support]; s, ty = support.to(device), sy.to(device)
            assert not torch.isin(support, query).any()
            assert torch.all(torch.bincount(sy, minlength=classes) == 5)
            for seed in seeds:
                for method in plan["methods"]:
                    folder = output/"5"/str(split)/str(seed)/method
                    folder.mkdir(parents=True, exist_ok=True)
                    metadata = dict(plan_sha256=plan["plan_sha256"], dataset=job["dataset"], family=job["family"],
                        checkpoint_id=job["id"], checkpoint_sha256=job["checkpoint_sha256"], pretrain_seed=job["pretrain_seed"],
                        graph_sha256=graph_hash, split_policy_sha256=digest({"pool": tensor_hash(pool), "query": tensor_hash(query)}),
                        shots=5, split_seed=split, train_seed=seed, method=method,
                        support_labels_sha256=tensor_hash(sy), smoke=args.smoke)
                    signature = digest(metadata)
                    if (folder/"metrics.json").exists():
                        saved = json.loads((folder/"metrics.json").read_text())
                        assert saved["signature"] == signature
                        for filename, key in [("state.pt", "state_sha256"), ("selection.json", "selection_sha256"),
                                              ("predictions.npz", "predictions_sha256")]:
                            assert sha_file(folder/filename) == saved[key]
                        completed += 1
                        continue
                    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
                    trace = {}
                    if method == "F100":
                        z, prompt, trace = fit_f100(u, r, adj, model, edges, s, ty, classes, seed, steps, filters)
                        gradients = prompt["optimizer_gradients"]
                        assert gradients.shape == (steps, classes, 128) and torch.isfinite(gradients).all()
                        assert float(gradients.abs().max()) > 0 and torch.count_nonzero(prompt["tokens"]) > 0
                        assert not trace["kd_active"] and trace["trainable_parameters"] == classes*128
                        with torch.no_grad():
                            center = z.mean(0, keepdim=True); centered = z-center
                            scores = old["logits"](centered, s, ty, classes)
                        state = dict(prompt=prompt, raw_embedding=z.cpu(), center=center.cpu(),
                                     centered_embedding=centered.cpu(), scale=scale.cpu())
                        chosen = dict(lr=.01, weight_decay=0, steps=steps, adam_epsilon=1e-8)
                        selection = dict(selection="fixed_historical_F100_recipe", config=chosen, query_labels_used=False,
                            M2=False, M3=False, KD=False, prompt_penalty=.001, M1_temperature=.1)
                        audit = dict(backbone_frozen=True, backbone_before=before, backbone_after=state_hash(model),
                            query_labels_received=False, prompt_parameters=classes*128, head_parameters=0,
                            backbone_parameters=0, trainable_parameters=classes*128,
                            prompt_gradient_max=float(gradients.abs().max()), gradient_steps=steps,
                            M2_active=False, M3_active=False, kd_active=False, adam_actual_epsilon=1e-8)
                    elif method == "FrozenPrototypeCentered":
                        with torch.no_grad():
                            z = model(x, edges); center = z.mean(0, keepdim=True)
                            scores = old["logits"](z-center, s, ty, classes)
                        state = dict(raw_embedding=z.cpu(), center=center.cpu())
                        chosen = {}; selection = dict(selection="parameter_free", trials=[], query_labels_used=False)
                        audit = dict(backbone_frozen=True, backbone_before=before, backbone_after=before,
                            query_labels_received=False, prompt_parameters=0, head_parameters=0,
                            backbone_parameters=0, trainable_parameters=0)
                    else:
                        chosen, selection = select_config(method, model, x, edges, s, ty, classes, config, seed)
                        scores, state, audit = fit_predict(method, model, x, edges, s, ty, classes, chosen, seed)
                    assert before == state_hash(model) and all(p.grad is None for p in model.parameters())
                    assert not selection["query_labels_used"] and not audit["query_labels_received"]
                    state.update(support=support, support_labels=sy, query=query, sealed_scores=scores.cpu())
                    torch.save(state, folder/"state.pt"); save_json(folder/"selection.json", selection)
                    seal = seal_predictions(folder, scores, support, query.to(device), dict(metadata, signature=signature))
                    # First query-label indexing for scoring is strictly after prediction sealing.
                    metrics = evaluate_sealed(folder, labels[query])
                    torch.cuda.synchronize(device)
                    row = dict(metadata, signature=signature, predictions_sha256=seal["predictions_sha256"],
                        state_sha256=sha_file(folder/"state.pt"), selection_sha256=sha_file(folder/"selection.json"),
                        support_sha256=seal["support_sha256"], query_sha256=seal["query_sha256"],
                        evaluation_labels_sha256=tensor_hash(labels[query]), selected_config=chosen, metrics=metrics,
                        audit=audit, trace=trace, total_seconds=time.perf_counter()-started,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved(device),
                        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=str(device)))
                    save_json(folder/"metrics.json", row); completed += 1
                    print(json.dumps(dict(job=job["id"], complete=completed,
                        expected=len(splits)*len(seeds)*len(plan["methods"]), method=method,
                        seconds=row["total_seconds"], peak_cuda_bytes=row["peak_cuda_bytes"])), flush=True)
                    del state, scores
        save_json(output/f"completion_shard_{args.shard}.json", dict(units=completed, smoke=args.smoke, plan_sha256=plan["plan_sha256"]))
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
