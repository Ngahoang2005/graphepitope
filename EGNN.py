# EGNN.py
import torch
import torch.nn as nn

class EGNN(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, edge_dim=51, 
                 dropout=0.2, residual=True, normalize=True, epsilon=1e-8):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.normalize = normalize
        self.epsilon = epsilon

        # 1. Edge MLP (Sâu hơn, nhiều tham số hơn)
        in_edge = in_dim * 2 + edge_dim + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, in_dim) # Bắt buộc ra in_dim
        )

        # 2. Coord MLP (Dùng để dịch chuyển toạ độ 3D)
        layer = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            layer
        )

        # 3. Node MLP (Sâu hơn, update đặc trưng node)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + in_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, in_dim) # Bắt buộc ra in_dim để cộng residual
        )

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        diff = coord[row] - coord[col]
        dist2 = (diff**2).sum(dim=-1, keepdim=True)
        dist2 = torch.clamp(dist2, min=self.epsilon, max=100.0)
        
        if self.normalize:
            norm = (dist2.sqrt().detach() + self.epsilon)
            diff = diff / norm
        return dist2, diff

    def forward(self, h, edge_index, edge_attr, pos):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, pos)

        # Edge update
        e_in = [h[row], h[col], edge_attr, radial]
        e = self.edge_mlp(torch.cat(e_in, dim=-1))

        # Coordinate update
        coord_update = self.coord_mlp(e)
        coord_update = torch.clamp(coord_update, -1.0, 1.0)
        trans = coord_diff * coord_update
        
        agg_coord = torch.zeros_like(pos)
        agg_coord.index_add_(0, row, trans)
        pos = pos + agg_coord

        # Node update
        agg_node = torch.zeros_like(h)
        agg_node.index_add_(0, row, e)
        
        x_in = torch.cat([h, agg_node], dim=-1)
        h_new = self.node_mlp(x_in)
        
        # Residual connection an toàn
        if self.residual and h_new.shape[-1] == h.shape[-1]:
            h_new = h + h_new

        return h_new, pos