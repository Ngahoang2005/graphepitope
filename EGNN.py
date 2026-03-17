# EGNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EGNNConv

class EGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = EGNNConv(in_dim, hidden_dim, edge_dim=edge_dim, aggr='mean')
        self.conv2 = EGNNConv(hidden_dim, hidden_dim, edge_dim=edge_dim, aggr='mean')

    def forward(self, x, edge_index, edge_attr, pos=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr, pos) + x   # residual connection
        x = self.conv2(x, edge_index, edge_attr, pos)
        return x