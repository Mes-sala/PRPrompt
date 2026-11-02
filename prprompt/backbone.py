"""Two-layer frozen GCN with ProG-compatible checkpoint keys."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class TwoLayerGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        self.conv_layers = nn.ModuleList([GCNConv(input_dim, hidden_dim),
                                          GCNConv(hidden_dim, output_dim)])

    def forward(self, x, edge_index, batch=None):
        x = self.conv_layers[1](F.leaky_relu(self.conv_layers[0](x, edge_index)), edge_index)
        return x if batch is None else global_mean_pool(x, batch.long())

    @classmethod
    def from_checkpoint(cls, path, input_dim, hidden_dim=128, device='cpu'):
        model = cls(input_dim, hidden_dim, hidden_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        return model.eval().requires_grad_(False)

def normalized_adjacency(edge_index, nodes, dtype):
    indices, weights = gcn_norm(edge_index, num_nodes=nodes, add_self_loops=True, dtype=dtype)
    return torch.sparse_coo_tensor(indices.flip(0), weights, (nodes, nodes)).coalesce()
