#EGAT.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class AE(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, dropout = 0., bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden, bias=bias),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_out, bias=bias),
            nn.LayerNorm(dim_out),
        )
    def forward(self, x):
        return self.net(x)
class GVP(nn.Module):
    """Lõi Geometric Vector Perceptron"""
    def __init__(self, in_dims, out_dims, h_dim=None):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.h_dim = h_dim or max(1, self.vo)
        
        self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
        self.ws = nn.Linear(self.h_dim + self.si, self.so)
        self.wv = nn.Linear(self.h_dim, self.vo, bias=False)

    def forward(self, x):
        s, v = x
        v_h = self.wh(v)
        v_n = torch.norm(v_h, dim=-1)
        s_out = self.ws(torch.cat([s, v_n], dim=-1))
        v_out = self.wv(v_h)
        s_out = F.relu(s_out)
        return s_out, v_out

class DenseGVPConv(nn.Module):
    """Lớp chập (Convolution) dựa trên GVP cho ma trận Dense (vì edge_attr của mày là Dense)"""
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.message = GVP((in_features * 2, 1), (out_features, 1))
        self.update = GVP((out_features, 1), (out_features, 1))

    def forward(self, h, edge_attr, v):
        N = h.shape[0]
        # Tạo message scalar (Nối node nguồn và node đích)
        h_exp1 = h.unsqueeze(1).expand(N, N, -1)
        h_exp2 = h.unsqueeze(0).expand(N, N, -1)
        s_msg = torch.cat([h_exp1, h_exp2], dim=-1) # [N, N, 2*in_features]
        
        # Tạo message vector (Khoảng cách giữa các node)
        v_msg = v.unsqueeze(1).expand(N, N, -1, -1) - v.unsqueeze(0).expand(N, N, -1, -1)
        
        s_m, v_m = self.message((s_msg, v_msg))
        
        # Tính mask từ edge_attr (edge_attr có shape [efeat, N, N])
        mask = (edge_attr.sum(dim=0) > 0).float() 
        
        # Gộp thông tin (Pooling)
        s_pool = (s_m * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        v_pool = (v_m * mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        
        s_out, v_out = self.update((s_pool, v_pool))
        s_out = self.dropout(s_out)
        
        return s_out, v_out
class EGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_attr):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        e = e*edge_attr
        zero_vec = -9e15*torch.ones_like(e)
        e = torch.where(edge_attr > 0, e, zero_vec)
        e = F.softmax(e, dim=1)
        e = F.dropout(e, self.dropout, training=self.training)
        
        h_prime=[]
        for i in range(edge_attr.shape[0]):
            h_prime.append(torch.matmul(e[i],Wh))
        
        if self.concat:
            h_prime = torch.cat(h_prime,dim=1)
        else:
            h_prime = torch.stack(h_prime,dim=0).mean(0)
        return F.elu(h_prime),e

    #compute attention coefficient
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class EGAT(nn.Module):
    def __init__(self, nfeat, nhid, efeat, dropout=0.2, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = DenseGVPConv(nfeat, nhid, dropout)
        self.conv2 = DenseGVPConv(nhid, nfeat, dropout)
    def forward(self, x, edge_attr):
        x_cut=x
        v = torch.randn(x.shape[0], 1, 3, device=x.device) 
        
        # Chạy qua 2 lớp GVP
        s, v = self.conv1(x, edge_attr, v)
        s, v = self.conv2(s, edge_attr, v)
        
        # Trả về scalar (s) + Residual Connection giống hệt form cũ
        return s + x_cut, edge_attr