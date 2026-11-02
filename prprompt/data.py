"""Explicit TU loading and historical ProG task sampling; separate from routing."""
from pathlib import Path
import torch
from torch_geometric.datasets import TUDataset


def load_tu(name, root):
    dataset = TUDataset(root=str(Path(root)/'TUDataset'), name=name, use_node_attr=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        dataset = dataset.shuffle()
    graphs = list(dataset)
    labels = torch.tensor([int(g.y.item()) for g in graphs])
    return graphs, labels, dataset.num_classes


def sample_graph_task(labels, shot, task):
    labels = torch.as_tensor(labels).cpu().long()
    classes = int(labels.max())+1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026102300+task)
        order = torch.randperm(len(labels)); query = order[:int(.8*len(labels))]
        remaining = order[len(query):]
        si = []
        for c in range(classes):
            candidates = [i for i in remaining if int(labels[i]) == c]
            if len(candidates) < shot:
                raise ValueError(f'Class {c} has fewer than {shot} support candidates')
            si.extend(candidates[:shot])
        support = torch.tensor(si)[torch.randperm(len(si))]
    return support, query
