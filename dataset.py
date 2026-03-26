import os
import torch
import warnings
import argparse
import numpy as np
import pickle as pk
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset

warnings.simplefilter('ignore')

class PDB(Dataset):
    def __init__(self, mode='train', fold=-1, root='./data/Epitope3D'):
        self.root = root
        assert mode in ['train', 'val', 'test']
        
        if mode in ['train', 'val']:
            with open(f'{self.root}/train.pkl', 'rb') as f:
                self.samples = pk.load(f)
        else:
            with open(f'{self.root}/test.pkl', 'rb') as f:
                self.samples = pk.load(f)
        
        self.data = []
        idx = np.load(f'{self.root}/cross-validation.npy')
        cv = 10
        inter = len(idx) // cv
        ex = len(idx) % cv
        
        if mode == 'train':
            order = []
            for i in range(cv):
                if i == fold: continue
                order += list(idx[i*inter : (i+1)*inter + ex*(i==cv-1)])
        elif mode == 'val':
            order = list(idx[fold*inter : (fold+1)*inter + ex*(fold==cv-1)])
        else:
            order = list(range(len(self.samples)))
            
        order.sort()
        tbar = tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            # Nạp đặc trưng Node (ESM-C và SaProt)
            self.samples[i].load_feat(self.root)
            self.samples[i].load_saprot(self.root)
            # Nạp tọa độ CA (File .graph giờ chỉ chứa 'pos')
            self.samples[i].load_graph(self.root) 
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        f = seq.feat
        d = seq.saprot
        pos = seq.pos # Tọa độ [N, 3] từ file .graph
        
        # Cắt BOS/EOS của ESM-C để khớp với SaProt/PDB
        if f.shape[0] == d.shape[0] + 2:
            f = f[1:-1, :]
            
        min_len = min(f.shape[0], d.shape[0], pos.shape[0])
        
        f = f[:min_len, :]
        d = d[:min_len, :]
        pos = pos[:min_len, :]
        label = seq.label[:min_len]
        
        feat = torch.cat([f, d], dim=1)

        return {
            'x': feat,
            'pos': pos,
            'y': label
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/Epitope3D')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_csv', type=str, default='epitope3d_dataset_180_train.csv')
    parser.add_argument('--test_csv', type=str, default='epitope3d_dataset_20_test.csv')
    args = parser.parse_args()
    
    root = args.root
    device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
    
    os.makedirs(f'{root}/PDB', exist_ok=True)
    os.makedirs(f'{root}/purePDB', exist_ok=True)
    os.makedirs(f'{root}/feat', exist_ok=True)
    os.makedirs(f'{root}/saprot', exist_ok=True)
    os.makedirs(f'{root}/graph', exist_ok=True)

    from esm.sdk import client
    token = '5zPJa56XnPf91N4L8yWdMQ'
    
    print("[INFO] Connecting to Forge API...")
    model = client(
        model="esmc-6b-2024-12", 
        url="https://forge.evolutionaryscale.ai", 
        token=token
    )
    
    
    print("[INFO] Processing Trainset...")
    trainset = initial_epitope3D(args.train_csv, root, model, device)
        
    print("[INFO] Processing Testset...")
    testset = initial_epitope3D(args.test_csv, root, model, device)
        
    # Lọc các chuỗi quá dài hoặc không có epitope
    trainset = [i for i in trainset if len(i) < 1024 and getattr(i, 'label', None) is not None and i.label.sum() > 0]
    testset = [i for i in testset if len(i) < 1024 and getattr(i, 'label', None) is not None and i.label.sum() > 0]

    np.random.seed(42) 
    idx = np.random.permutation(len(trainset))
    
    with open(f'{root}/train.pkl', 'wb') as f:
        pk.dump(trainset, f)
    with open(f'{root}/test.pkl', 'wb') as f:
        pk.dump(testset, f)

    np.save(f'{root}/cross-validation.npy', idx)
    print(f"Done! Train: {len(trainset)}, Test: {len(testset)}")