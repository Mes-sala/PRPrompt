"""GraphTOP operators ported from xbfu/GraphTOP at 7c662da0.

The upstream task.py (not the paper's entropy sign / sparsity norm) is
authoritative. See docs/SUPT_GRAPHTOP_PORT.md for provenance and boundaries.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, is_undirected, k_hop_subgraph, scatter


class Rewiring(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # Upstream explicitly reinitializes weights only; preserve bias init.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def annealing_temperature(epoch, epochs):
    if epochs < 1 or not 1 <= epoch <= epochs:
        raise ValueError("GraphTOP requires 1 <= epoch <= epochs")
    return 0.97 * (1 - epoch / epochs) + 0.03


def sample_edges(prob, tau, training, generator=None):
    """Upstream two-uniform relaxation; inference is Bernoulli, not threshold."""
    eps = 1e-10

    def rand():
        return torch.rand(prob.shape, dtype=prob.dtype, device=prob.device, generator=generator)

    if not training:
        return (prob > rand()).to(prob.dtype)
    if tau <= 0:
        raise ValueError("tau must be positive")
    u0, u1 = rand(), rand()
    noise = -torch.log(torch.log(u1 + eps) / torch.log(u0 + eps) + eps)
    return torch.sigmoid((torch.log(prob + eps) - torch.log(1 - prob + eps) + noise) / tau)


def topology_losses(prob_all, batch):
    """Positive global candidate entropy; per-graph squared density target 0.5."""
    prob = prob_all[batch.neighbor_mask]
    eps = 1e-10
    entropy = (
        (-(prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))).mean()
        if prob.numel()
        else prob_all.sum() * 0
    )
    count = batch.ptr[1:] - batch.ptr[:-1] - 1
    degree = scatter(prob_all, batch.batch, dim=0, reduce="sum").view(-1)
    degree = degree - prob_all[batch.target_node_index].view(-1)
    valid = count > 0
    # An isolated target has no rewiring decisions: no entropy/density penalty.
    sparsity = (
        ((degree[valid] / count[valid] - 0.5) ** 2).mean() if valid.any() else degree.sum() * 0
    )
    return entropy, sparsity


class TopologyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in {"node_idx", "original_idx"}:
            return 0  # Global graph IDs must never be batch-offset.
        return super().__inc__(key, value, *args, **kwargs)


def get_topology_subgraphs(data, node_list, num_hops=2):
    """Upstream 2-hop ego graph and paired target-star candidates, label-free.

    Only the target label is attached after constructing topology. Unlike the
    generic ProG induced-graph builder, no padding/truncation is performed.
    """
    if num_hops < 1:
        raise ValueError("GraphTOP num_hops must be positive")
    data = data.cpu()
    edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
    if not is_undirected(edge_index, num_nodes=data.num_nodes):
        raise ValueError("GraphTOP official implementation requires undirected graphs")
    if getattr(data, "edge_weight", None) is not None and not torch.all(data.edge_weight == 1):
        raise ValueError("GraphTOP port requires an unweighted input graph")
    graphs = []
    for node in torch.as_tensor(node_list).view(-1).tolist():
        subset, edges, mapping, _ = k_hop_subgraph(
            node, num_hops, edge_index, relabel_nodes=True, num_nodes=data.num_nodes
        )
        target = int(mapping.item())
        n = subset.numel()
        neighbors = torch.arange(n)[torch.arange(n) != target]
        pairs = torch.stack((torch.full_like(neighbors, target), neighbors))
        star = torch.stack((pairs, pairs.flip(0)), dim=-1).reshape(2, -1)
        # Preserve all non-target edges and any existing self-loop.
        keep = ((edges[0] != target) & (edges[1] != target)) | (edges[0] == edges[1])
        existing = set(map(tuple, edges.t().tolist()))
        weights = torch.tensor([float((target, int(j)) in existing) for j in neighbors])
        graph = TopologyData(
            x=data.x[subset].clone(),
            edge_index=torch.cat((star, edges[:, keep]), dim=1),
            edge_weight=torch.cat((weights.repeat_interleave(2), torch.ones(int(keep.sum())))),
            trainable_edge=torch.cat(
                (
                    torch.ones(star.size(1), dtype=torch.bool),
                    torch.zeros(int(keep.sum()), dtype=torch.bool),
                )
            ),
            neighbor_mask=torch.arange(n) != target,
            target_node_index=torch.tensor([target]),
            original_idx=torch.tensor([node]),
            node_idx=subset,
            y=data.y.view(-1)[node].view(1),
        )
        graphs.append(graph)
    return graphs


def prompted_edge_weights(prompt, embeddings, batch, tau, training, generator=None):
    pair_features = embeddings[batch.original_idx][batch.batch] + embeddings[batch.node_idx]
    prob_all = torch.sigmoid(prompt(pair_features) / 0.2)
    prob = prob_all[batch.neighbor_mask]
    sampled = sample_edges(prob, tau, training, generator=generator)
    weights = batch.edge_weight.clone()
    weights[batch.trainable_edge] = sampled.view(-1).repeat_interleave(2)
    return weights, prob_all
