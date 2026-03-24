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

        # 🔥 ADD: LayerNorm để ổn định feature
        self.node_norm = nn.LayerNorm(in_dim)

        # 1. Edge MLP
        in_edge = in_dim * 2 + edge_dim + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, in_dim),
            nn.Tanh()   # 🔥 ADD: chặn magnitude
        )

        # 2. Coord MLP
        layer = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.coord_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            layer
        )

        # 3. Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + in_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, in_dim)
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

        # ================= EDGE =================
        e_in = [h[row], h[col], edge_attr, radial]
        e = self.edge_mlp(torch.cat(e_in, dim=-1))

        # ================= COORD =================
        coord_update = self.coord_mlp(e)

        # 🔥 FIX 1: dùng tanh thay vì clamp
        coord_update = torch.tanh(coord_update)

        # 🔥 FIX 2: scale cực kỳ quan trọng
        trans = coord_diff * coord_update * 0.1

        agg_coord = torch.zeros_like(pos)
        agg_coord.index_add_(0, row, trans)

        pos = pos + agg_coord

        # ================= NODE =================
        agg_node = torch.zeros_like(h)
        agg_node.index_add_(0, row, e)
        
        x_in = torch.cat([h, agg_node], dim=-1)
        h_new = self.node_mlp(x_in)

        # 🔥 FIX 3: giảm residual strength
        if self.residual and h_new.shape[-1] == h.shape[-1]:
            h_new = h + 0.5 * h_new

        # 🔥 FIX 4: normalize feature
        h_new = self.node_norm(h_new)

        return h_new, pos