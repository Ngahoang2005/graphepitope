#EGNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter  # Comment out to avoid dependency
# from torch_geometric.nn import InstanceNorm  # Comment out

class EGNN(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, edge_dim=51, 
                 dropout=0.2, residual=True, attention=False, 
                 normalize=False, coords_agg='mean', tanh=False, 
                 ffn=False, batch_norm=False):
        super().__init__()
        self.in_dim = in_dim          # 512
        self.hidden_dim = hidden_dim  # 256
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.ffn = ffn
        self.batch_norm = batch_norm

    def __init__(self, in_dim=512, hidden_dim=256, edge_dim=51, 
                 dropout=0.2, residual=True, attention=False, 
                 normalize=False, coords_agg='sum', tanh=False,  # Changed to 'sum' for index_add_
                 ffn=False, batch_norm=False):
        super().__init__()
        self.in_dim = in_dim          # 512
        self.hidden_dim = hidden_dim  # 256
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.ffn = ffn
        self.batch_norm = batch_norm  # Set to False

        # === SỬA Ở ĐÂY: Edge MLP phải output ra in_dim (512) ===
        in_edge = in_dim * 2 + edge_dim + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, in_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

        # Coord MLP
        layer = nn.Linear(in_dim, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_blocks = [nn.Linear(in_dim, in_dim), nn.SiLU(),
                        nn.Dropout(dropout), layer]
        if tanh: coord_blocks.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_blocks)

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + in_dim, hidden_dim),   # 512 + 512 = 1024 → 256
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Batch normalization (commented out)
        # if batch_norm:
        #     self.norm_node = InstanceNorm(hidden_dim, affine=True)
        #     self.norm_coord = InstanceNorm(3, affine=True)

        # FFN
        if ffn:
            self.ff1 = nn.Linear(hidden_dim, hidden_dim * 2)
            self.ff2 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.act_ff = nn.SiLU()
            self.drop_ff = nn.Dropout(dropout)
            # if batch_norm:
            #     self.norm_ff1 = InstanceNorm(hidden_dim, affine=True)
            #     self.norm_ff2 = InstanceNorm(hidden_dim, affine=True)

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        diff = coord[row] - coord[col]
        dist2 = (diff**2).sum(dim=-1, keepdim=True)
        
        # Clamp distance to prevent extreme values
        dist2 = torch.clamp(dist2, min=self.epsilon, max=100.0)
        
        if self.normalize:
            norm = (dist2.sqrt().detach() + self.epsilon)
            diff = diff / norm
            # Check for NaN/Inf in normalized diff
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
        return dist2, diff

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.drop_ff(self.act_ff(self.ff1(x)))
        return self.ff2(x)

    def forward(self, h, edge_index, edge_attr, pos):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, pos)

        # -- edge features --
        e_in = [h[row], h[col], edge_attr, radial]
        e = torch.cat(e_in, dim=-1)
        e = self.edge_mlp(e)
        if self.attention:
            att = self.att_mlp(e)
            e = e * att

        # -- coordinate update --
        coord_update = self.coord_mlp(e)  # [E,1]
        # Clamp coordinate updates to prevent explosion
        coord_update = torch.clamp(coord_update, -1.0, 1.0)
        trans = coord_diff * coord_update  # [E,3]
        
        # Check for NaN/Inf in coordinate updates
        trans = torch.where(torch.isfinite(trans), trans, torch.zeros_like(trans))
        
        agg_coord = torch.zeros_like(pos)
        agg_coord.index_add_(0, row, trans)
        pos = pos + agg_coord
        
        # Check for NaN/Inf in final coordinates
        pos = torch.where(torch.isfinite(pos), pos, torch.zeros_like(pos))
        
        # if self.batch_norm:
        #     # For single graph, batch is all zeros
        #     batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        #     pos = self.norm_coord(pos, batch)

        # -- node update --
        agg_node = torch.zeros_like(h)
        agg_node.index_add_(0, row, e)
        x_in = torch.cat([h, agg_node], dim=-1)
        h_new = self.node_mlp(x_in)
        # if self.batch_norm:
        #     h_new = self.norm_node(h_new, batch)
        if self.residual and h_new.shape[-1] == h.shape[-1]:
            h_new = h + h_new

        # -- optional FFN --
        if self.ffn:
            # if self.batch_norm:
            #     h_new = self.norm_ff1(h_new, batch)
            h_new = h_new + self._ff_block(h_new)
            # if self.batch_norm:
            #     h_new = self.norm_ff2(h_new, batch)

        return h_new, pos