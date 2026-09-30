"""Small utilities shared by the local masked-graph pretraining ports.

The original projects save a backbone checkpoint after pretraining.  ProG
uses the same convention, so these helpers deliberately keep the decoder and
method-specific state out of the exported checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

from prompt_graph.data import load4node
from prompt_graph.defines import NODE_TASKS


def resolve_node_data(dataset_name: str, data, device: torch.device):
    """Load a ProG node graph or use an injected graph for tests/experiments."""

    if data is None:
        if dataset_name not in NODE_TASKS:
            raise ValueError(
                f"The local masked pretraining ports currently require a node dataset; "
                f"got {dataset_name!r}."
            )
        data, input_dim, out_dim = load4node(dataset_name)
    else:
        if data.x is None or data.edge_index is None:
            raise ValueError("pretraining data must contain x and edge_index")
        input_dim = int(data.x.size(-1))
        out_dim = int(getattr(data, "num_classes", 0) or 0)
    return data.to(device), input_dim, out_dim


def write_provenance_sidecar(
    checkpoint_path,
    *,
    method: str,
    dataset: str,
    gnn: str,
    hidden_dim: int,
    metadata=None,
) -> None:
    """Write a provenance sidecar (``<checkpoint>.json``) for a saved checkpoint.

    Shared by the local masked ports (via ``save_encoder_checkpoint``) and the
    paper-scope pretrain methods (whose ``pretrain()`` saves a bare state_dict).
    The critical field is ``checkpoint_sha256``; the v4 runner compares it
    against the file hash to decide ``metadata_recorded_adapted`` vs
    ``legacy_unverified``.
    """
    checkpoint_path = Path(checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    record = dict(schema="prog-pretrain-provenance-v1", method=method,
                  variant="ProG-adapted", source_parity="not_fully_verified",
                  dataset=dataset, gnn=gnn, hidden_dim=hidden_dim,
                  checkpoint_sha256=digest, **(metadata or {}))
    sidecar = Path(str(checkpoint_path) + ".json")
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, default=str) + "\n")
    temporary.replace(sidecar)


def save_encoder_checkpoint(
    encoder: torch.nn.Module,
    dataset_name: str,
    method_name: str,
    gnn_type: str,
    hid_dim: int,
    output_path=None,
    metadata=None,
) -> str:
    """Save an encoder state dict using ProG's existing filename convention."""

    if output_path is None:
        folder = Path("Experiment") / "pre_trained_model" / dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / f"{method_name}.{gnn_type}.{hid_dim}hidden_dim.pth"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), output_path)
    write_provenance_sidecar(
        output_path,
        method=method_name,
        dataset=dataset_name,
        gnn=gnn_type,
        hidden_dim=hid_dim,
        metadata=metadata,
    )
    return str(output_path)


def validate_rate(name: str, value: float, *, allow_zero: bool = True) -> float:
    value = float(value)
    lower = 0.0 if allow_zero else 1e-8
    if not lower <= value <= 1.0:
        raise ValueError(f"{name} must be in [{lower}, 1], got {value}")
    return value


def safe_index_subset(num_nodes: int, rate: float, device: torch.device) -> Tensor:
    """Return a non-empty random subset for a valid masking rate."""

    count = int(num_nodes * rate)
    if rate > 0 and num_nodes > 0:
        count = max(1, count)
    count = min(count, num_nodes)
    return torch.randperm(num_nodes, device=device)[:count]


def pretrain_metadata(task, completed_epochs, loss):
    digest = hashlib.sha256()
    for name in ("x", "edge_index"):
        value = getattr(task.data, name).detach().cpu().contiguous()
        digest.update(str((name, value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    scalars = {k: v for k, v in vars(task).items()
               if not k.startswith("_") and isinstance(v, (str, int, float, bool, tuple))}
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.glob("*.py"))}
    return dict(input_hash=digest.hexdigest(), labels_used=False,
                initial_torch_seed=torch.initial_seed(), requested_epochs=task.epochs,
                completed_epochs=completed_epochs, selection="last_after_training_loss_patience_10",
                final_training_loss=loss, scalar_configuration=scalars, source_files=sources,
                optimizer_groups=[dict(lr=g["lr"], weight_decay=g.get("weight_decay", 0))
                                  for g in task.optimizer.param_groups])
