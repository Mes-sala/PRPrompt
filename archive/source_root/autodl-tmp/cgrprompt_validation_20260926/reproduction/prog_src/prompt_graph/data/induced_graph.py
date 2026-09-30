"""Version 3 induced graphs: topology-only node selection, label-independent cache."""
import hashlib
import json
import os
import pickle

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from prompt_graph.utils import get_logger

logger = get_logger(__name__)


def _input_identity(data, smallest_size, largest_size):
    digest = hashlib.sha256(f'topology-v3:{smallest_size}:{largest_size}'.encode())
    for value in (data.x, data.edge_index):
        v = value.detach().cpu().contiguous()
        digest.update(str((v.dtype, tuple(v.shape))).encode())
        digest.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _build_single_induced_graph(data, center_idx, smallest_size, largest_size,
                                initial_hop=2, max_hop=5, label=None, candidate_mask=None):
    # candidate_mask remains a compatibility argument; labels/masks NEVER select nodes.
    if not 1 <= smallest_size <= largest_size or not 0 <= initial_hop <= max_hop:
        raise ValueError('Invalid subgraph size/hops')
    subset = torch.tensor([center_idx], device=data.edge_index.device)
    for hops in range(initial_hop, max_hop + 1):
        subset, _, _, _ = k_hop_subgraph(
            center_idx, hops, data.edge_index, num_nodes=data.num_nodes
        )
        subset = torch.unique(torch.cat([subset, subset.new_tensor([center_idx])]))
        if subset.numel() >= smallest_size:
            break
    if subset.numel() > largest_size:
        other = subset[subset != center_idx].sort().values
        generator = torch.Generator().manual_seed(1729 + int(center_idx))
        take = torch.randperm(other.numel(), generator=generator)[:largest_size-1].to(other.device)
        subset = torch.cat([other[take], other.new_tensor([center_idx])]).sort().values
    # Small connected components remain variable-sized; no label-guided padding.
    edges, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    target = int(data.y[center_idx]) if label is None else int(label)
    return Data(x=data.x[subset], edge_index=edges, y=target,
                node_ids=subset, index=int(center_idx))


def induced_graphs(data, smallest_size=10, largest_size=30):
    return [_build_single_induced_graph(data, i, smallest_size, largest_size)
            for i in range(data.num_nodes)]


def induced_graph_cache_path(dir_path, smallest_size, largest_size, leak_safe=True,
                             input_identity=None):
    # Never reuse legacy .pkl or _v2.pkl. The mask is no longer an input dependency.
    suffix = '_' + input_identity if input_identity else ''
    return os.path.join(dir_path, f'induced_graph_min{smallest_size}_max{largest_size}_v3{suffix}.pkl')


def split_induced_graphs(data, dir_path, device, smallest_size=10, largest_size=30,
                         train_mask=None, input_identity=None):
    graphs = [g.cpu() for g in induced_graphs(data, smallest_size, largest_size)]
    identity = _input_identity(data, smallest_size, largest_size)
    if input_identity is not None and input_identity != identity:
        raise ValueError('Cache input identity mismatch')
    path = induced_graph_cache_path(dir_path, smallest_size, largest_size, input_identity=input_identity)
    os.makedirs(dir_path, exist_ok=True)
    # Atomic replace prevents readers from observing an incomplete pickle.
    temporary = path + f'.{os.getpid()}.tmp'
    with open(temporary, 'wb') as handle:
        pickle.dump({'version': 3, 'input_identity': identity, 'graphs': graphs}, handle)
    os.replace(temporary, path)
    with open(path + '.json', 'w') as handle:
        json.dump({'version': 3, 'input_identity': identity, 'label_dependent_sampling': False,
                   'smallest_size_is_target_not_guarantee': True, 'num_graphs': len(graphs)}, handle, indent=2)
    return path


def load_induced_graphs(dataset_name, data, device, smallest_size=100, largest_size=300):
    from prompt_graph.utils import induced_graph_dir
    directory = str(induced_graph_dir(dataset_name))
    identity = _input_identity(data, smallest_size, largest_size)
    path = induced_graph_cache_path(directory, smallest_size, largest_size, input_identity=identity)
    if not os.path.exists(path):
        split_induced_graphs(data, directory, device, smallest_size, largest_size, input_identity=identity)
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    if payload.get('version') != 3 or payload.get('input_identity') != identity:
        raise ValueError('Invalid induced graph cache')
    graphs = payload['graphs']
    if len(graphs) != data.num_nodes:
        raise ValueError('Incomplete graph cache')
    for i, graph in enumerate(graphs):
        if int(graph.index) != i:
            raise ValueError('Cached center order mismatch')
        # Labels are targets only and may change without invalidating graph inputs.
        graph.y = data.y[i].detach().cpu().view(1)
    return [graph.to(device) for graph in graphs]
