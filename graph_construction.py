#graph_construction.py
import torch
import numpy as np

ID={
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

def calcPROgraph(coord):
    """
    Trong cấu trúc HGNN mới, file này không cần tính toán cạnh (edges).
    HGNN sẽ dùng tọa độ để tính toán on-the-fly.
    Hàm này đảm bảo tọa độ được trả về đúng định dạng Tensor.
    """
    pos = torch.as_tensor(coord, dtype=torch.float32)
    
    return {'pos': pos}