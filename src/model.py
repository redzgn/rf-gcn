import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
