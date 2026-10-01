"""Load unmodified v1 function ASTs, without executing historical launch code."""
import ast
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F

REFERENCE = Path(__file__).resolve().parent / "reference"
FUNCTIONS = ["thash", "ridge_teacher", "smooth_probs", "rectified_teacher", "make_filters",
             "logits", "temperature_prompt", "optimizer_prompt"]


def extract(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert set(names) == {n.name for n in nodes}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def load():
    namespace = {"torch": torch, "F": F, "hashlib": hashlib}
    extract(REFERENCE/"act.py", ["act"], namespace)
    extract(REFERENCE/"gate79.py", ["temperature_gate"], namespace)
    extract(REFERENCE/"run_round83_graft.py", FUNCTIONS, namespace)
    return namespace


def fit_f100(u, r, adjacency, model, edges, support, support_labels, classes, seed, steps, filters):
    old = load()
    return old["optimizer_prompt"](u, r, adjacency, model, edges, support, support_labels,
                                   classes, seed, "token_ridge_kd10plain", steps, filters,
                                   "none", 0.1, "raw", "adam", 0.01)
