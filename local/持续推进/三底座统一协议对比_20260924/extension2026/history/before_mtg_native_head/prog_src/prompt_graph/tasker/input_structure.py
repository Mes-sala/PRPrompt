"""Isolated fixed-budget NodeTask runner, preserving the existing v3 splits."""

import hashlib

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from prompt_graph.data.load4data import audit_node_indices
from prompt_graph.prompt.GraphTOP import get_topology_subgraphs
from prompt_graph.utils import get_logger, sample_dir

from .strategy import TaskContext, get_strategy

logger = get_logger(__name__)
INPUT_STRUCTURE_PROMPTS = {"SUPT-soft", "SUPT-hard", "GraphTOP"}


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def input_structure_options(args):
    defaults = {
        "supt_clusters": 5,
        "supt_ratio": 0.4,
        "supt_orth_loss": False,
        "supt_softmax": False,
        "supt_branch": "chem",
        "graphtop_hops": 2,
        "graphtop_lambda_e": 2.0,
        "graphtop_lambda_s": 10.0,
    }
    return {k: getattr(args, k, v) for k, v in defaults.items()}


def run_input_structure_node(task):
    if task.epochs < 1 or task.task_num < 1:
        raise ValueError("epochs and task_num must be positive")
    rows, losses = [], []
    task.port_diagnostics = []
    # Build graph structure once, independently of labels/splits.
    if task.prompt_type == "GraphTOP":
        if task.gnn_type != "GCN":
            raise ValueError(
                "GraphTOP requires GCN: other ProG convolutions do not share edge-weight semantics"
            )
        graph_cache = {}
    else:
        if task.graphs_list is None:
            raise ValueError("SUPT NodeTask requires ProG v3 label-free induced graphs")
        graph_cache = {int(g.index): g for g in task.graphs_list}

    for fold in range(1, task.task_num + 1):
        folder = task.fold_dir(fold)
        train_idx = (
            torch.load(folder / "train_idx.pt", map_location="cpu", weights_only=True)
            .long()
            .view(-1)
        )
        test_idx = (
            torch.load(folder / "test_idx.pt", map_location="cpu", weights_only=True)
            .long()
            .view(-1)
        )
        audit_node_indices(train_idx, test_idx, task.data.y.cpu(), task.shot_num, task.output_dim)
        for name, idx in [("train", train_idx), ("test", test_idx)]:
            saved = torch.load(
                folder / f"{name}_labels.pt", map_location="cpu", weights_only=True
            ).view(-1)
            if not torch.equal(saved, task.data.y.cpu().view(-1)[idx]):
                raise ValueError("Saved labels do not match dataset")
        if task.prompt_type == "GraphTOP":
            missing = [i for i in torch.cat((train_idx, test_idx)).tolist() if i not in graph_cache]
            graphs = get_topology_subgraphs(
                task.data.clone(), missing, task.prompt_options.get("graphtop_hops", 2)
            )
            graph_cache.update(zip(missing, graphs))
        train_graphs = [graph_cache[int(i)] for i in train_idx]
        test_graphs = [graph_cache[int(i)] for i in test_idx]
        if task.protocol_options:
            task.batch_size = len(train_idx)
            task.epochs = task.protocol_options["support_passes"]
            from prompt_graph.utils import seed_everything
            seed_everything(task.protocol_options["train_seed"] + fold)
        train_loader = DataLoader(train_graphs, batch_size=task.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_graphs, batch_size=task.batch_size * (5 if task.prompt_type == "GraphTOP" else 1)
        )
        task.initialize_gnn()
        if task.protocol_options:
            seed_everything(task.protocol_options["train_seed"] + 10000 + fold)
        task.answering = torch.nn.Sequential(torch.nn.Linear(task.hid_dim, task.output_dim)).to(
            task.device
        )
        if task.protocol_options:
            seed_everything(task.protocol_options["train_seed"] + 20000 + fold)
        task.initialize_prompt()
        task.initialize_optimizer()
        if task.prompt_options.get("disable_prompt", False):
            task.prompt.requires_grad_(False)
            task.optimizer = torch.optim.Adam(task.answering.parameters(), lr=task.lr, weight_decay=task.wd)
        audit = None
        if task.protocol_options:
            from .audit import FoldAudit
            audit = FoldAudit(task, fold)
        before = state_hash(task.gnn)
        extra = {
            "disable_prompt": task.prompt_options.get("disable_prompt", False),
            "epochs": task.epochs,
            "lambda_e": task.prompt_options.get("graphtop_lambda_e", 2.0),
            "lambda_s": task.prompt_options.get("graphtop_lambda_s", 10.0),
        }
        if task.prompt_type == "GraphTOP":
            with torch.no_grad():
                extra["embeddings"] = task.gnn(
                    task.data.x.to(task.device), task.data.edge_index.to(task.device)
                ).detach()
        ctx = TaskContext(
            gnn=task.gnn,
            prompt=task.prompt,
            answering=task.answering,
            optimizer=task.optimizer,
            device=task.device,
            extra=extra,
        )
        strategy = get_strategy(task.prompt_type)()
        if task.protocol_options:
            seed_everything(task.protocol_options["train_seed"] + 30000 + fold)
        for epoch in range(1, task.epochs + 1):
            extra["epoch"] = epoch
            loss = strategy.train_epoch(ctx, train_loader)
            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"{task.prompt_type} nonfinite loss at fold {fold}, epoch {epoch}"
                )
            logger.info(
                "%s fold %d epoch %d/%d loss %.6f", task.prompt_type, fold, epoch, task.epochs, loss
            )
        after = state_hash(task.gnn)
        if before != after:
            raise RuntimeError("Frozen encoder state changed during prompt tuning")
        if task.protocol_options:
            seed_everything(task.protocol_options["inference_seed"] + fold)
        rows.append(strategy.evaluate(ctx, test_loader))
        losses.append(loss)
        if audit is not None:
            task.run_diagnostics.append(audit.finish(task.epochs, rows[-1], len(train_idx), len(test_idx)))
        task.port_diagnostics.append(
            {
                "fold": fold,
                "encoder_hash_before": before,
                "encoder_hash_after": after,
                "epochs": task.epochs,
                "selection": "fixed_last",
                "support": len(train_idx),
                "query": len(test_idx),
                "prompt_options": task.prompt_options,
                "metrics": rows[-1],
            }
        )
    values = np.asarray(rows)
    result = [float(np.mean(losses))]
    for column in values.T:
        result.extend((float(np.mean(column)), float(np.std(column))))
    return tuple(result)
