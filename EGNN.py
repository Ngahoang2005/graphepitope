#EGNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EGNN(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, edge_dim=51, 
                 dropout=0.2, residual=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.epsilon = 1e-8

        # Edge MLP → output ra đúng hidden_dim để aggregate
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim + 1, hidden_dim),   # +1 là radial
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Coord MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, edge_index, edge_attr, pos):
        row, col = edge_index

        # Tính radial distance
        diff = pos[row] - pos[col]
        radial = (diff ** 2).sum(dim=-1, keepdim=True).sqrt() + self.epsilon

        # Edge feature
        edge_in = torch.cat([h[row], h[col], edge_attr, radial], dim=-1)
        e = self.edge_mlp(edge_in)                     # ← giờ output = hidden_dim

        # Coordinate update (equivariant)
        coord_update = self.coord_mlp(e)
        coord_update = torch.clamp(coord_update, -1.0, 1.0)
        trans = coord_update * diff

        # Aggregate coordinate
        agg_coord = torch.zeros_like(pos)
        agg_coord.index_add_(0, row, trans)
        pos = pos + agg_coord

        # Node update
        agg_node = torch.zeros_like(h)
        agg_node.index_add_(0, row, e)                 # ← giờ khớp chiều (512)

        h_new = self.node_mlp(torch.cat([h, agg_node], dim=-1))

        if self.residual and h_new.shape[-1] == h.shape[-1]:
            h_new = h + h_new

        return h_new, pos