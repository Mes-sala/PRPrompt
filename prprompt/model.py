"""Prompt operators. No labels other than the supplied support labels enter here."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class TwoLayerGCN(nn.Module):
    """State keys and activation match the archived two-layer ProG GCN."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            GCNConv(input_dim, hidden_dim), GCNConv(hidden_dim, output_dim)
        ])

    def forward(self, x, edge_index):
        return self.conv_layers[1](F.leaky_relu(self.conv_layers[0](x, edge_index)), edge_index)


def normalized_adjacency(edge_index, nodes, dtype):
    indices, weights = gcn_norm(edge_index, num_nodes=nodes, add_self_loops=True, dtype=dtype)
    return torch.sparse_coo_tensor(indices.flip(0), weights, (nodes, nodes)).coalesce()


@torch.no_grad()
def attribute_features(x, adjacency):
    """Fixed two-hop attributes; full-graph centering uses no labels."""
    r = torch.sparse.mm(adjacency, torch.sparse.mm(adjacency, x))
    return F.normalize(r - r.mean(0, keepdim=True), dim=1)


def prototypes(features, support, support_labels, classes):
    counts = torch.bincount(support_labels, minlength=classes)
    if len(counts) != classes or (counts == 0).any():
        raise ValueError("Every task class must occur in support")
    sums = torch.zeros(classes, features.shape[1], device=features.device, dtype=features.dtype)
    sums.index_add_(0, support_labels, features[support])
    return F.normalize(sums / counts[:, None], dim=1)


@torch.no_grad()
def prototype_route(features, adjacency, support, support_labels, classes, temperature=0.1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    centers = prototypes(features, support, support_labels, classes)
    probabilities = (features @ centers.T / temperature).softmax(-1)
    route = torch.sparse.mm(adjacency, probabilities - probabilities.mean(0, keepdim=True))
    return route, probabilities


class PrototypeRoutedPrompt(nn.Module):
    def __init__(self, route, hidden, amplitude=10.0):
        super().__init__()
        self.register_buffer("route", route.detach())
        self.register_buffer("scale", hidden.std(0, unbiased=False).clamp_min(1e-6).detach())
        self.tokens = nn.Parameter(hidden.new_zeros(route.shape[1], hidden.shape[1]))
        self.amplitude = amplitude

    def forward(self, hidden):
        return hidden + self.amplitude * (self.route @ self.tokens.tanh()) * self.scale


class FeaturePrompt(nn.Module):
    """GPF / GPF-plus equations; full-graph node protocol, not original GIN experiments."""
    def __init__(self, input_dim, tokens=1, attentive=False):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(tokens, input_dim))
        nn.init.xavier_uniform_(self.basis)
        self.router = nn.Linear(input_dim, tokens) if attentive else None

    def forward(self, x):
        return x + (self.basis if self.router is None else self.router(x).softmax(-1) @ self.basis)


class ScratchEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kind):
        super().__init__()
        conv = {"ScratchGAT": GATConv, "ScratchSAGE": SAGEConv}[kind]
        self.layers = nn.ModuleList([conv(input_dim, hidden_dim), conv(hidden_dim, hidden_dim)])

    def forward(self, x, edges):
        return self.layers[1](F.leaky_relu(self.layers[0](x, edges)), edges)
