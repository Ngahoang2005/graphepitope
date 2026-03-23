import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import SE3Transformer

class SE3TransformerWrapper(nn.Module):
    """
    Wrapper cho SE3Transformer từ torch_geometric.
    SE(3)-Transformer: Equivariant attention cho graphs 3D.
    """
    def __init__(self, in_dim=512, hidden_dim=256, edge_dim=51, 
                 dropout=0.2, num_layers=2, num_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        # SE3Transformer từ torch_geometric
        self.se3_transformer = SE3Transformer(
            in_channels=in_dim,  # Input node features
            out_channels=hidden_dim,  # Output node features
            edge_channels=edge_dim,  # Edge features
            num_layers=num_layers,  # Số layers
            num_heads=num_heads,  # Attention heads
            channels_div=2,  # Channel division
            attn_dropout=dropout,
            act=nn.SiLU(),
            num_degrees=3,  # Degrees for spherical harmonics
            num_basis=10  # Number of basis functions
        )
        
        # Project input nếu cần
        if in_dim != hidden_dim:
            self.input_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.input_proj = None

    def forward(self, h, edge_index, edge_attr, pos):
        """
        h: [num_nodes, in_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim]
        pos: [num_nodes, 3]
        """
        if self.input_proj is not None:
            h = self.input_proj(h)
        
        # SE3Transformer expects pos as [num_nodes, 3]
        # edge_attr as [num_edges, edge_dim]
        h_out, pos_out = self.se3_transformer(h, pos, edge_index, edge_attr)
        
        return h_out, pos_out