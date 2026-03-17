# graph_construction.py
import torch
import numpy as np

ID = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

def calcPROgraph(seq, coord, dseq=3, dr=10, dlong=5, k=10):
    """
    Xây graph cho Equivariant GNN (EGNN)
    - Trả về: edge_index, edge_attr (51 chiều), pos (tọa độ Cα)
    - Không còn sparse adj/edge nữa
    """
    nodes = coord.shape[0]
    
    # Tạo ma trận tạm để xây edge (giữ nguyên logic cũ)
    adj = torch.zeros((nodes, nodes))                    # chỉ dùng để tìm index
    E   = torch.zeros((nodes, nodes, 21*2 + 2*dseq + 3)) # 51 đặc trưng cạnh

    dist = torch.cdist(coord, coord, p=2)
    knn  = dist.argsort(1)[:, 1:k+1]

    for i in range(nodes):
        for j in range(nodes):
            if i == j:                  
                continue
            not_edge = True
            dij_seq = abs(i - j)

            # Sequential edge
            if dij_seq < dseq:
                E[i][j][41 + i - j + dseq] = 1
                not_edge = False

            # Radius edge
            if dist[i][j] < dr and dij_seq >= dlong:
                E[i][j][41 + 2*dseq] = 1
                not_edge = False

            # KNN edge
            if j in knn[i] and dij_seq >= dlong:
                E[i][j][42 + 2*dseq] = 1
                not_edge = False

            if not_edge:
                continue

            # Đánh dấu edge tồn tại
            adj[i][j] = 1

            # One-hot amino acid + distance features (giữ nguyên)
            E[i][j][ID.get(seq[i], 20)] = 1
            E[i][j][21 + ID.get(seq[j], 20)] = 1
            E[i][j][43 + 2*dseq] = dij_seq
            E[i][j][44 + 2*dseq] = dist[i][j]

    # ===================== PHẦN MỚI: CHUYỂN SANG ĐỊNH DẠNG PyG =====================
    edge_index = adj.nonzero(as_tuple=False).T.long()      # shape: [2, num_edges]
    edge_attr  = E[edge_index[0], edge_index[1]]           # shape: [num_edges, 51]

    pos = coord.clone()                                    # shape: [nodes, 3]

    return {
        'edge_index': edge_index,   # ← bắt buộc cho EGNN
        'edge_attr':  edge_attr,    # ← 51 đặc trưng cạnh (giữ nguyên như cũ)
        'pos':        pos           # ← tọa độ Cα cho Equivariant
    }