"""Scientific audit from state, sealed predictions and frozen inputs; no fitting."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
import numpy as np
import torch
from torch.nn import functional as F
from prprompt.model import TwoLayerGCN, normalized_adjacency, prototypes
from prprompt.protocol import save_json, sha_file, sample_support
from prprompt.train import tensor_hash, state_hash
from experiments.v1_f100.historical import load
from experiments.f100_multidata.run import read_plan, digest
from metric_utils import numpy_metrics


@torch.no_grad()
def baseline_scores(method, state, model, x, edges, support, sy, classes, config):
    # Reconstruct equations directly from stored parameters, without fit_predict.
    if "model" in state:
        model.load_state_dict(state["model"], strict=True)
    if method == "FrozenPrototypeCentered":
        z = model(x, edges)
        return load()["logits"](z-z.mean(0, keepdim=True), support, sy, classes)
    if method in ["FrozenLinear", "FullFineTune", "FrozenPrototype"]:
        z = model(x, edges)
    elif method in ["GPF", "GPFPlus"]:
        prompt = {k: v.to(x.device) for k, v in state["prompt"].items()}
        delta = prompt["basis"] if method == "GPF" else F.linear(x, prompt["router.weight"], prompt["router.bias"]).softmax(-1)@prompt["basis"]
        z = model(x+delta, edges)
    elif method == "GraphPromptDiag":
        z = model(x, edges)*state["prompt"]["weight"].to(x.device)
    elif method == "PRPrompt":
        p = {k: v.to(x.device) for k, v in state["prompt"].items()}
        hidden = model.conv_layers[0](x, edges)
        z = model.conv_layers[1](F.leaky_relu(hidden+config["amplitude"]*(p["route"]@p["tokens"].tanh())*p["scale"]), edges)
    else:
        raise ValueError(method)
    z = F.normalize(z, dim=1)
    if "head" in state:
        return F.linear(z, state["head"]["weight"].to(x.device), state["head"]["bias"].to(x.device))
    return z@prototypes(z, support, sy, classes).T/.2


def audit(args):
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plan = read_plan(args.plan); job = plan["jobs"][args.job_index]; old = load(); device = torch.device(args.device)
    spec = plan["datasets"][job["dataset"]]
    assert sha_file(spec["snapshot"]) == spec["snapshot_sha256"]
    data = torch.load(spec["snapshot"], map_location="cpu", weights_only=True)
    x, edges, labels, pool, query = (data[k] for k in ["x", "edges", "labels", "pool", "query"])
    assert digest({"x": tensor_hash(x), "edges": tensor_hash(edges)}) == spec["graph_sha256"]
    classes = spec["classes"]; x, edges = x.to(device), edges.to(device)
    original = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
    assert sha_file(job["checkpoint"]) == job["checkpoint_sha256"]
    model = TwoLayerGCN(job["input_dim"], 128, 128).to(device).eval(); model.requires_grad_(False)
    model.load_state_dict(original); original_hash = state_hash(model)
    adj = normalized_adjacency(edges, len(x), x.dtype)
    with torch.no_grad():
        u = model.conv_layers[0](x, edges)
        r = torch.sparse.mm(adj, torch.sparse.mm(adj, x))
    prefix = "smoke" if args.smoke else "results"
    root = args.plan.parent/prefix/job["id"]
    splits = plan["split_seeds"][:1] if args.smoke else plan["split_seeds"]
    seeds = plan["train_seeds"][:1] if args.smoke else plan["train_seeds"]
    expected = {(s, t, m) for s in splits for t in seeds for m in plan["methods"]}
    seen = set(); pairing = {}; details = []
    max_metric_error = max_gate_error = max_mixed = max_replay_error = 0.
    for path in sorted(root.glob("5/*/*/*/metrics.json")):
        row = json.loads(path.read_text()); identity = (row["split_seed"], row["train_seed"], row["method"])
        assert identity in expected and identity not in seen; seen.add(identity)
        assert row["plan_sha256"] == plan["plan_sha256"] and row["smoke"] == args.smoke
        assert row["signature"] == digest({k: row[k] for k in ["plan_sha256", "dataset", "family", "checkpoint_id",
            "checkpoint_sha256", "pretrain_seed", "graph_sha256", "split_policy_sha256", "shots", "split_seed",
            "train_seed", "method", "support_labels_sha256", "smoke"]})
        for filename, key in [("state.pt", "state_sha256"), ("selection.json", "selection_sha256"), ("predictions.npz", "predictions_sha256")]:
            assert sha_file(path.parent/filename) == row[key]
        seal = json.loads((path.parent/"seal.json").read_text())
        for key in ["plan_sha256", "signature", "predictions_sha256", "support_sha256", "query_sha256"]:
            assert seal[key] == row[key]
        state = torch.load(path.parent/"state.pt", map_location="cpu", weights_only=True)
        s, sy, q = (state[k] for k in ["support", "support_labels", "query"])
        assert torch.equal(s, sample_support(pool, labels[pool], classes, 5, row["split_seed"]))
        assert torch.equal(q, query) and torch.equal(sy, labels[s]) and not torch.isin(s, q).any()
        for value, field in [(s, "support_sha256"), (q, "query_sha256"), (sy, "support_labels_sha256"), (labels[q], "evaluation_labels_sha256")]:
            assert tensor_hash(value) == row[field]
        matched = tuple(row[k] for k in ["graph_sha256", "support_sha256", "query_sha256", "support_labels_sha256"])
        if row["split_seed"] in pairing:
            assert pairing[row["split_seed"]] == matched
        pairing[row["split_seed"]] = matched
        with np.load(path.parent/"predictions.npz") as sealed:
            p = sealed["probabilities"].copy()
            assert np.array_equal(sealed["support"], s.numpy()) and np.array_equal(sealed["query"], q.numpy())
        replay = state["sealed_scores"][q].double().softmax(-1).numpy()
        assert np.array_equal(replay, p), "Serialization replay must be exact"
        got = numpy_metrics(labels[q].numpy(), p)
        for key, value in got.items():
            error = abs(value-row["metrics"][key]); max_metric_error = max(max_metric_error, error)
            assert error < plan["audit_rules"]["metric_atol"]
        a = row["audit"]; assert not a["query_labels_received"] and a["backbone_before"] == original_hash
        selection = json.loads((path.parent/"selection.json").read_text()); assert not selection["query_labels_used"]
        model.load_state_dict(original)
        si, ty, qi = s.to(device), sy.to(device), q.to(device)
        diagnostic = dict(unit=str(path.relative_to(args.plan.parent)), method=row["method"])
        with torch.no_grad():
            if row["method"] == "F100":
                assert selection["config"] == dict(lr=.01, weight_decay=0, steps=2 if args.smoke else 100, adam_epsilon=1e-8)
                assert not any(a[k] for k in ["M2_active", "M3_active", "kd_active"]) and a["head_parameters"] == 0
                prompt = state["prompt"]; g = prompt["optimizer_gradients"]
                assert g.shape == (selection["config"]["steps"], classes, 128) and torch.isfinite(g).all()
                assert float(g.abs().max()) == a["prompt_gradient_max"] > 0
                raw = old["rectified_teacher"](r-r.mean(0, keepdim=True), si, ty, classes, "rect_teacher_ref50", row["train_seed"])[0]
                gate, info = old["temperature_gate"](raw, adj, .1, "raw")
                gate_error = float((gate-prompt["gate"].to(device)).abs().max())
                max_gate_error = max(max_gate_error, gate_error); assert gate_error < 2e-6
                assert torch.allclose(info["probability"], prompt["gate_probability"].to(device), atol=2e-6, rtol=0)
                saved = state["raw_embedding"].to(device)
                assert torch.equal(saved-state["center"].to(device), state["centered_embedding"].to(device))
                scores = old["logits"](state["centered_embedding"].to(device), si, ty, classes)
                restored = scores[qi].cpu().double().softmax(-1).numpy()
                error = float(np.abs(restored-p).max()); max_replay_error = max(max_replay_error, error)
                assert error < 2e-6 and np.array_equal(restored.argmax(1), p.argmax(1))
                scale = u.std(0).clamp_min(1e-6)
                assert torch.allclose(scale, state["scale"].to(device), atol=1e-6, rtol=1e-6)
                z = model.conv_layers[1](F.leaky_relu(u+10*(prompt["gate"].to(device)@prompt["tokens"].to(device).tanh())*state["scale"].to(device)), edges)
                diff = (z-saved).abs(); ratio = float((diff/(2e-6+1e-6*saved.abs())).max())
                assert ratio < 1; max_mixed = max(max_mixed, ratio)
                diagnostic.update(raw_forward_max_error=float(diff.max()), mixed_forward_error_ratio=ratio)
                scores = old["logits"](z-z.mean(0, keepdim=True), si, ty, classes)
                zg = model.conv_layers[1](F.leaky_relu(u+10*(gate@prompt["tokens"].to(device).tanh())*scale), edges)
                rebuild = old["logits"](zg-zg.mean(0, keepdim=True), si, ty, classes)[qi].cpu().double().softmax(-1).numpy()
                diagnostic.update(rebuilt_gate_probability_error=float(np.abs(rebuild-p).max()),
                    rebuilt_gate_class_changes=int(np.sum(rebuild.argmax(1) != p.argmax(1))))
            else:
                if "trials" in selection and selection["trials"]:
                    assert len(selection["trials"]) == (1 if args.smoke else 4)
                    assert all(len(t["fold_ce"]) == 3 for t in selection["trials"])
                    best = min(selection["trials"], key=lambda t: t["mean_ce"])
                    assert best["config"] == row["selected_config"]
                scores = baseline_scores(row["method"], state, model, x, edges, si, ty, classes, row["selected_config"])
            rebuilt = scores[qi].cpu().double().softmax(-1).numpy()
            diagnostic.update(independent_state_probability_error=float(np.abs(rebuilt-p).max()),
                independent_state_score_error=float((scores.cpu()-state["sealed_scores"]).abs().max()),
                independent_state_class_changes=int(np.sum(rebuilt.argmax(1) != p.argmax(1))))
            assert state_hash(model) == a["backbone_after"]
        if row["method"] != "FullFineTune":
            assert a["backbone_frozen"] and a["backbone_after"] == original_hash
        else:
            assert not a["backbone_frozen"] and a["backbone_parameters"] > 0
        if row["method"] in ["F100", "GPF", "GPFPlus", "GraphPromptDiag", "PRPrompt"]:
            assert a["prompt_gradient_max"] > 0
        details.append(diagnostic)
    assert seen == expected, f"Missing {len(expected-seen)} cells"
    output = args.plan.parent/("smoke_analysis" if args.smoke else "analysis")
    save_json(output/(job["id"]+"_audit.json"), dict(expected=len(expected), present=len(seen), missing=[],
        plan_sha256=plan["plan_sha256"], checkpoint_id=job["id"], smoke=args.smoke,
        maximum_metric_error=max_metric_error, maximum_gate_error=max_gate_error,
        maximum_serialized_f100_readout_error=max_replay_error, maximum_mixed_forward_ratio=max_mixed,
        all_saved_scores_replay_exact=True, positive_prompt_gradients=True,
        matched_support_query_passed=True, details=details))
    print(json.dumps(dict(job=job["id"], audited=len(seen), smoke=args.smoke)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    audit(parser.parse_args())
