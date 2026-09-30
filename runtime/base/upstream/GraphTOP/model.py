import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data, edge_weight=None, pooling=False):
        assert pooling in ['mean', 'target', False]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if edge_weight is None:
            edge_weight = data.edge_weight

        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index, edge_weight)

        if pooling == 'mean':
            # Subgraph pooling to obtain the graph embeddings
            graph_emb = global_mean_pool(x2, batch.long())
            return graph_emb
        if pooling == 'target':
            # Extract the embedding of target nodes as the graph embeddings
            graph_emb = x2[data.target_node_index]
            return graph_emb

        return x2
