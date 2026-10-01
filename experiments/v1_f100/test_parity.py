"""Validate both historical Adam entry points and their actual frozen behavior."""
import argparse
import copy
import torch
from prprompt.model import TwoLayerGCN, normalized_adjacency
from prprompt.train import state_hash
from experiments.v1_f100.historical import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261001)
    x = torch.randn(79, 13, device=args.device)
    ids = torch.arange(len(x), device=args.device)
    edges = torch.stack([torch.cat([ids, ids]), torch.cat([(ids+1)%len(x), (ids-1)%len(x)])])
    model = TwoLayerGCN(13, 16, 16).to(args.device).eval()
    model.requires_grad_(False)
    before = state_hash(model)
    adjacency = normalized_adjacency(edges, len(x), x.dtype)
    with torch.no_grad():
        u = model.conv_layers[0](x, edges)
        r = torch.sparse.mm(adjacency, torch.sparse.mm(adjacency, x))
    s = torch.arange(15, device=args.device)
    sy = torch.arange(3, device=args.device).repeat_interleave(5)
    old = load()
    filters = old["make_filters"](x, edges)
    a, pa, ta = old["temperature_prompt"](u, r, adjacency, model, edges, s, sy, 3, 0,
                                          "token_ridge_kd10plain", 5, filters, "none", .1, "raw")
    b, pb, tb = old["optimizer_prompt"](u, r, adjacency, model, edges, s, sy, 3, 0,
                                        "token_ridge_kd10plain", 5, filters, "none", .1, "raw", "adam", .01)
    assert before == state_hash(model) and all(p.grad is None for p in model.parameters())
    assert not ta["kd_active"] and not tb["kd_active"] and tb["optimizer82"] == "adam"
    assert tb["trainable_parameters"] == 3*16 and pb["optimizer_gradients"].shape == (5, 3, 16)
    assert torch.count_nonzero(pb["tokens"])>0 and float(pb["optimizer_gradients"].abs().max())>0
    sa = old["logits"](a-a.mean(0, keepdim=True), s, sy, 3).softmax(-1)
    sb = old["logits"](b-b.mean(0, keepdim=True), s, sy, 3).softmax(-1)
    error = float((sa-sb).abs().max())
    if args.device == "cpu":
        assert torch.equal(pa["tokens"], pb["tokens"]) and torch.equal(sa, sb)
    else:
        assert error < 2e-6, error
    defaults = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=.01).defaults
    assert defaults["eps"] == 1e-8 and defaults["weight_decay"] == 0
    print(f"PASS v1 F100 Adam entry-point probability parity: {error:.9g}")
    print("PASS frozen backbone, prompt gradients, no KD, no M2/M3, Adam eps=1e-8 and weight_decay=0")


if __name__ == "__main__":
    main()
