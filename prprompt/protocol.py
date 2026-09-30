"""Matched support sampling, support-only selection, and sealed evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train import fit_predict, FIXED_METHODS, tensor_hash


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_dataset(name, root):
    from torch_geometric.datasets import Planetoid, Amazon, Actor, WebKB, WikipediaNetwork
    from torch_geometric.transforms import NormalizeFeatures
    root = Path(root)
    if name in {"Cora", "CiteSeer", "PubMed"}:
        data = Planetoid(str(root / "Planetoid"), name, transform=NormalizeFeatures())[0]
    elif name in {"Computers", "Photo"}:
        data = Amazon(str(root / "Amazon"), name, transform=NormalizeFeatures())[0]
    elif name == "Actor":
        data = Actor(str(root / "Actor"), transform=NormalizeFeatures())[0]
    elif name == "Cornell":
        data = WebKB(str(root / "Cornell"), "Cornell", transform=NormalizeFeatures())[0]
    elif name == "Chameleon":
        data = WikipediaNetwork(str(root / "WikipediaNetwork"), "chameleon", geom_gcn_preprocess=True,
                                transform=NormalizeFeatures())[0]
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    labels = data.y.cpu()
    if name in {"Computers", "Photo"}:
        generator = torch.Generator().manual_seed(20260930)
        pool = []
        for c in labels.unique(sorted=True):
            ids = (labels == c).nonzero().flatten()
            if len(ids) < 21:
                raise ValueError("Amazon split needs at least 21 nodes per class")
            pool.append(ids[torch.randperm(len(ids), generator=generator)[:20]])
        pool = torch.cat(pool)
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[pool] = False
        query = mask.nonzero().flatten()
    else:
        train = data.train_mask[:, 0] if data.train_mask.ndim == 2 else data.train_mask
        test = data.test_mask[:, 0] if data.test_mask.ndim == 2 else data.test_mask
        pool = train.nonzero().flatten().cpu()
        query = test.nonzero().flatten().cpu()
    if torch.isin(pool, query).any():
        raise ValueError("Support pool overlaps query")
    classes = int(labels[pool].max()) + 1
    if len(labels[pool].unique()) != classes:
        raise ValueError("Non-contiguous/missing task classes in support pool")
    return data.x.cpu(), data.edge_index.cpu(), labels, pool, query, classes


def sample_support(pool, pool_labels, classes, shots, seed):
    if shots < 1:
        raise ValueError("shots must be positive")
    rng = torch.Generator().manual_seed(seed)
    selected = []
    for c in range(classes):
        ids = pool[pool_labels == c]
        if len(ids) < shots:
            raise ValueError(f"Class {c} has only {len(ids)} eligible labels for {shots}-shot")
        selected.append(ids[torch.randperm(len(ids), generator=rng)[:shots]])
    return torch.cat(selected)


def support_folds(support, support_labels, classes):
    counts = torch.bincount(support_labels, minlength=classes)
    if counts.min() < 2:
        return []
    folds = []
    for offset in range(min(3, int(counts.min()))):
        valid_pos = torch.stack([(support_labels == c).nonzero().flatten()[offset] for c in range(classes)])
        train_mask = torch.ones(len(support), dtype=torch.bool, device=support.device)
        train_mask[valid_pos] = False
        folds.append((support[train_mask], support_labels[train_mask], support[valid_pos], support_labels[valid_pos]))
    return folds


def select_config(method, backbone, x, edges, support, support_labels, classes, config, seed):
    base = config["fit"]
    folds = support_folds(support, support_labels, classes)
    if method in FIXED_METHODS or not folds:
        return dict(base), {"selection": "parameter_free" if method in FIXED_METHODS else "one_shot_predeclared",
                            "trials": [], "query_labels_used": False}
    trials = []
    for lr in config["selection"]["learning_rates"]:
        for wd in config["selection"]["weight_decays"]:
            candidate = dict(base, lr=lr, weight_decay=wd)
            losses = []
            for fold, (train, ty, valid, vy) in enumerate(folds):
                scores, _, _ = fit_predict(method, backbone, x, edges, train, ty, classes, candidate, seed + fold)
                losses.append(float(F.cross_entropy(scores[valid], vy)))
            trials.append({"config": candidate, "fold_ce": losses, "mean_ce": float(np.mean(losses))})
    best = min(trials, key=lambda r: r["mean_ce"])
    return best["config"], {"selection": "support_only_stratified_leave_one_per_class",
                            "trials": trials, "query_labels_used": False}


def seal_predictions(directory, scores, support, query, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    probs = scores[query].detach().cpu().double().softmax(-1).numpy()
    if not np.isfinite(probs).all():
        raise ValueError("Invalid probabilities")
    output = directory / "predictions.npz"
    np.savez_compressed(output, probabilities=probs, support=support.cpu().numpy(), query=query.cpu().numpy())
    seal = dict(metadata, predictions_sha256=sha_file(output),
                support_sha256=tensor_hash(support), query_sha256=tensor_hash(query))
    save_json(directory / "seal.json", seal)
    return seal


def evaluate_sealed(directory, query_labels):
    """The sole scoring API that accepts query labels, after prediction file + seal exist."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
    directory = Path(directory)
    seal = json.loads((directory / "seal.json").read_text())
    if sha_file(directory / "predictions.npz") != seal["predictions_sha256"]:
        raise ValueError("Sealed predictions changed")
    with np.load(directory / "predictions.npz") as data:
        p = data["probabilities"]
    y = query_labels.cpu().numpy()
    if len(y) != len(p):
        raise ValueError("Query labels do not match sealed predictions")
    pred = p.argmax(1)
    classes = p.shape[1]
    result = {"accuracy": float(accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=list(range(classes)), average="macro", zero_division=0)),
              "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0))}
    # Explicit nulls when a query split omits a class; never fabricate multiclass AUC.
    if len(np.unique(y)) == classes:
        onehot = np.eye(classes)[y]
        result["roc_auc_macro"] = float(roc_auc_score(onehot, p, average="macro"))
        result["average_precision_macro"] = float(average_precision_score(onehot, p, average="macro"))
    else:
        result.update(roc_auc_macro=None, average_precision_macro=None)
    return result
