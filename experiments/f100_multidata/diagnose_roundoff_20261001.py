"""Fixed-unit repeated inference on both GPUs; no fitting or query labels."""
import argparse
import json
import sys
from pathlib import Path


def diagnose(args):
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import torch
    from torch.nn import functional as F
    from prprompt.model import TwoLayerGCN
    from prprompt.protocol import save_json, sha_file
    from prprompt.train import state_hash
    from experiments.v1_f100.historical import load
    from experiments.f100_multidata.run import read_plan
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plan = read_plan(args.workspace/"plan.json")
    job = next(j for j in plan["jobs"] if j["dataset"] == "Actor" and j["family"] == "Edgepred_GPPT")
    folder = args.workspace/"results"/job["id"]/"5/30093000/0/F100"
    state = torch.load(folder/"state.pt", map_location="cpu", weights_only=True)
    data = torch.load(plan["datasets"]["Actor"]["snapshot"], map_location="cpu", weights_only=True)
    with np.load(folder/"predictions.npz") as sealed:
        p = sealed["probabilities"].copy()
    old = load(); rows = []
    for gpu in [0, 1]:
        device = torch.device(f"cuda:{gpu}")
        model = TwoLayerGCN(job["input_dim"], 128, 128).to(device).eval(); model.requires_grad_(False)
        model.load_state_dict(torch.load(job["checkpoint"], map_location="cpu", weights_only=True))
        before = state_hash(model); x, edges = data["x"].to(device), data["edges"].to(device)
        gate, tokens, scale = (state["prompt"]["gate"].to(device), state["prompt"]["tokens"].to(device), state["scale"].to(device))
        saved = state["raw_embedding"].to(device)
        s, sy, q = (state[k].to(device) for k in ["support", "support_labels", "query"])
        first = None
        with torch.no_grad():
            for repeat in range(10):
                hidden = model.conv_layers[0](x, edges)
                z = model.conv_layers[1](F.leaky_relu(hidden+10*(gate@tokens.tanh())*scale), edges)
                prob = old["logits"](z-z.mean(0, keepdim=True), s, sy, 5)[q].cpu().double().softmax(-1).numpy()
                delta = (z-saved).abs()
                rows.append(dict(gpu=gpu, repeat=repeat, maximum_raw_error=float(delta.max()),
                    mixed_error_ratio=float((delta/(2e-6+1e-6*saved.abs())).max()),
                    maximum_probability_error=float(np.abs(prob-p).max()),
                    classification_changes=int(np.sum(prob.argmax(1) != p.argmax(1))),
                    maximum_raw_difference_from_first=float((z-first).abs().max()) if first is not None else 0.))
                if first is None:
                    first = z.clone()
            assert before == state_hash(model)
        del model, x, edges, saved, first, z
        torch.cuda.empty_cache()
    output = dict(unit=str(folder.relative_to(args.workspace)), repetitions_per_gpu=10, query_labels_received=False,
        checkpoint_sha256=sha_file(job["checkpoint"]), state_sha256=sha_file(folder/"state.pt"),
        classification_changes=sum(r["classification_changes"] for r in rows),
        maximum_probability_error=max(r["maximum_probability_error"] for r in rows),
        maximum_raw_repeat_jitter=max(r["maximum_raw_difference_from_first"] for r in rows),
        predeclared_mixed_failures=sum(r["mixed_error_ratio"] >= 1 for r in rows),
        repeats=rows, note="Diagnostic only. All repetitions retained; no rerun-until-pass, no altered thresholds or predictions.")
    save_json(args.workspace/"audit_failure_evidence/repeated_forward_diagnostic.json", output)
    print(json.dumps({k: v for k, v in output.items() if k != "repeats"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    diagnose(parser.parse_args())
