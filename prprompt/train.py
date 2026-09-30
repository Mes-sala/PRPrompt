"""Training API deliberately accepts neither validation labels nor query labels."""
import copy
import hashlib
import time
import torch
from torch import nn
from torch.nn import functional as F
from .model import (TwoLayerGCN, ScratchEncoder, FeaturePrompt, PrototypeRoutedPrompt,
                    normalized_adjacency, attribute_features, prototypes, prototype_route)

MAIN_METHODS = ["FrozenLinear", "FrozenPrototype", "FullFineTune", "ScratchGCN",
                "ScratchGAT", "ScratchSAGE", "MLP", "GPF", "GPFPlus",
                "GraphPromptDiag", "AttributePrototype", "AttributeRidge", "PRPrompt"]
DIAGNOSTIC_METHODS = ["SharedHidden", "LearnedRouterHidden", "RandomFrozenPRPrompt"]
FIXED_METHODS = {"FrozenPrototype", "AttributePrototype", "AttributeRidge"}


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((str(t.dtype), tuple(t.shape))).encode() + t.numpy().tobytes()).hexdigest()


def state_hash(module):
    return hashlib.sha256("".join(k + tensor_hash(v) for k, v in sorted(module.state_dict().items())).encode()).hexdigest()


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_predict(method, backbone, x, edge_index, support, support_labels, classes,
                config, seed=0):
    """Return all-node scores, model state, and audits. Callers choose evaluation nodes.

    Fit sees x/A for the full transductive graph and labels ONLY for support.
    Backbone is copied, so one method cannot modify another method's checkpoint.
    """
    if method not in MAIN_METHODS + DIAGNOSTIC_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if config["steps"] < 1 or len(support) != len(support_labels):
        raise ValueError("Positive steps and aligned support labels required")
    if support.unique().numel() != support.numel():
        raise ValueError("Support indices must be unique")
    seed_all(seed)
    started = time.perf_counter()
    model = copy.deepcopy(backbone).to(x.device).eval()
    model.requires_grad_(False)
    if method in {"ScratchGCN", "RandomFrozenPRPrompt"}:
        model = TwoLayerGCN(x.shape[1], backbone.conv_layers[0].out_channels,
                            backbone.conv_layers[1].out_channels).to(x.device).eval()
        model.requires_grad_(method == "ScratchGCN")
    if method == "FullFineTune":
        model.requires_grad_(True)
    if method in {"ScratchGAT", "ScratchSAGE"}:
        model = ScratchEncoder(x.shape[1], backbone.conv_layers[1].out_channels, method).to(x.device)
    before = state_hash(model)
    frozen = method not in {"FullFineTune", "ScratchGCN", "ScratchGAT", "ScratchSAGE"}
    hidden_dim = backbone.conv_layers[0].out_channels
    out_dim = backbone.conv_layers[1].out_channels
    adjacency = normalized_adjacency(edge_index, len(x), x.dtype)
    with torch.no_grad():
        attrs = attribute_features(x, adjacency) if method in {
            "PRPrompt", "RandomFrozenPRPrompt", "SharedHidden", "LearnedRouterHidden",
            "AttributePrototype", "AttributeRidge"} else None
        z0 = model(x, edge_index).detach()
    if method in FIXED_METHODS:
        z = F.normalize(z0, dim=1) if method == "FrozenPrototype" else attrs
        if method == "AttributeRidge":
            zs = z[support].double()
            targets = F.one_hot(support_labels, classes).double()
            dual = torch.linalg.solve(zs @ zs.T + torch.eye(len(support), device=x.device, dtype=torch.float64), targets)
            scores = (z @ z[support].T) @ dual.float()
        else:
            scores = z @ prototypes(z, support, support_labels, classes).T / 0.2
        return scores.detach(), {}, {"method": method, "trainable_parameters": 0,
            "prompt_parameters": 0, "head_parameters": 0, "backbone_parameters": 0,
            "task_model_parameters": 0,
            "backbone_frozen": True, "backbone_before": before, "backbone_after": state_hash(model),
            "query_labels_received": False, "fit_seconds": time.perf_counter() - started}

    prompt = None
    route_probabilities = None
    base_hidden = None
    if method in {"PRPrompt", "RandomFrozenPRPrompt", "SharedHidden", "LearnedRouterHidden"}:
        with torch.no_grad():
            base_hidden = model.conv_layers[0](x, edge_index).detach()
            route, route_probabilities = prototype_route(attrs, adjacency, support, support_labels,
                                                         classes, config.get("temperature", 0.1))
        if method == "SharedHidden":
            route = torch.ones(len(x), 1, device=x.device)
            route_probabilities = None
        prompt = PrototypeRoutedPrompt(route, base_hidden, config.get("amplitude", 10.0)).to(x.device)
        if method == "LearnedRouterHidden":
            # Same injection position and token bank; the router is additionally learned.
            prompt.learned_router = nn.Linear(attrs.shape[1], classes).to(x.device)
    elif method in {"GPF", "GPFPlus"}:
        prompt = FeaturePrompt(x.shape[1], 1 if method == "GPF" else config.get("gpf_tokens", 20), method == "GPFPlus").to(x.device)
    elif method == "GraphPromptDiag":
        prompt = nn.Module()
        prompt.weight = nn.Parameter(torch.empty(out_dim, device=x.device))
        nn.init.uniform_(prompt.weight, -((6 / (out_dim + 1)) ** 0.5), (6 / (out_dim + 1)) ** 0.5)
    elif method == "MLP":
        prompt = nn.Sequential(nn.Linear(x.shape[1], hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, out_dim)).to(x.device)

    # Every linear-head method starts from the same head draw, independent of prompt RNG use.
    seed_all(seed + 1_000_003)
    head = None if method == "GraphPromptDiag" else nn.Linear(out_dim, classes).to(x.device)
    params = [p for p in model.parameters() if p.requires_grad]
    if prompt is not None:
        params += list(prompt.parameters())
    if head is not None:
        params += list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])

    def embedding():
        if method in {"PRPrompt", "RandomFrozenPRPrompt", "SharedHidden", "LearnedRouterHidden"}:
            if method == "LearnedRouterHidden":
                probabilities = prompt.learned_router(attrs).softmax(-1)
                route = torch.sparse.mm(adjacency, probabilities - probabilities.mean(0, keepdim=True))
                u = base_hidden + prompt.amplitude * (route @ prompt.tokens.tanh()) * prompt.scale
            else:
                u = prompt(base_hidden)
            return model.conv_layers[1](F.leaky_relu(u), edge_index)
        if method in {"GPF", "GPFPlus"}:
            return model(prompt(x), edge_index)
        if method == "GraphPromptDiag":
            return z0 * prompt.weight
        if method == "MLP":
            return prompt(x)
        if method == "FrozenLinear":
            return z0
        return model(x, edge_index)

    def predict():
        z = F.normalize(embedding(), dim=1)
        if head is None:
            return z @ prototypes(z, support, support_labels, classes).T / 0.2
        return head(z)

    noop_error = None
    if method in {"PRPrompt", "RandomFrozenPRPrompt", "SharedHidden", "LearnedRouterHidden"}:
        with torch.no_grad():
            zero_embedding = embedding()
            noop_error = float((zero_embedding - z0).abs().max())
            # CUDA scatter reduction order can differ between two equivalent forwards.
            atol = 5e-5 if x.device.type == "cuda" else 1e-6
            if not torch.allclose(zero_embedding, z0, atol=atol, rtol=1e-5):
                raise RuntimeError(f"Zero prompt changed the backbone: {noop_error}")
    first_loss = last_loss = None
    prompt_gradient_max = 0.0
    for step in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        scores = predict()
        loss = F.cross_entropy(scores[support], support_labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite support loss")
        loss.backward()
        if frozen and any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen backbone received gradients")
        for p in params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError("Non-finite gradient")
        if prompt is not None:
            norm = sum(float(p.grad.detach().square().sum()) for p in prompt.parameters() if p.grad is not None) ** 0.5
            prompt_gradient_max = max(prompt_gradient_max, norm)
        optimizer.step()
        if step == 0:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
    with torch.no_grad():
        scores = predict().detach()
    after = state_hash(model)
    if frozen and before != after:
        raise RuntimeError("Frozen checkpoint was modified")
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    for name, module in [("prompt", prompt), ("head", head)]:
        if module is not None:
            state[name] = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    audit = {"method": method, "trainable_parameters": sum(p.numel() for p in params),
             "prompt_parameters": sum(p.numel() for p in prompt.parameters()) if prompt is not None and method != "MLP" else 0,
             "task_model_parameters": sum(p.numel() for p in prompt.parameters()) if method == "MLP" else 0,
             "head_parameters": sum(p.numel() for p in head.parameters()) if head is not None else 0,
             "backbone_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "backbone_frozen": frozen, "backbone_before": before, "backbone_after": after,
             "query_labels_received": False, "support_label_hash": tensor_hash(support_labels),
             "support_index_hash": tensor_hash(support), "zero_prompt_max_error": noop_error,
             "prompt_gradient_max": prompt_gradient_max, "first_support_loss": first_loss,
             "last_support_loss": last_loss, "fit_seconds": time.perf_counter() - started}
    if method == "LearnedRouterHidden":
        with torch.no_grad():
            route_probabilities = prompt.learned_router(attrs).softmax(-1)
    if route_probabilities is not None:
        audit["route_hash"] = tensor_hash(route_probabilities)
        audit["route_entropy"] = float(-(route_probabilities * route_probabilities.clamp_min(1e-12).log()).sum(-1).mean())
    return scores, state, audit
