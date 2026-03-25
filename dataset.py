#dataset.py

import os
import esm
import esm.sdk
from esm.sdk import client
import torch
import warnings
import argparse
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader,Dataset
warnings.simplefilter('ignore')


class PDB(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/Epitope3D',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            #self.samples[i].load_dssp(self.root)
            self.samples[i].load_esm3(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq = self.data[idx]
        
        # --- FIX LỖI SIZE MISMATCH TẠI ĐÂY ---
        f = seq.feat
        e = seq.esm3
        
        # Cắt BOS/EOS của ESM-C (646 -> 644)
        if f.shape[0] == e.shape[0] + 2:
            f = f[1:-1, :]
            
        # Đảm bảo độ dài khớp tuyệt đối (đề phòng lỗi file PDB)
        min_len = min(f.shape[0], e.shape[0])
        f = f[:min_len, :]
        e = e[:min_len, :]
        
        # Ghép feature
        feat = torch.cat([f, e], dim=1)
        # -------------------------------------

        return {
            'feat': feat,
            'label': seq.label[:min_len], 
            'adj': seq.adj,
            'edge': seq.edge if hasattr(seq, 'edge') else None,
        }
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/Epitope3D', help='dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    parser.add_argument('--train_csv', type=str, default='epitope3d_dataset_180_train.csv', 
                        help='Tên file CSV train cho tập epitope3d')
    parser.add_argument('--test_csv', type=str, default='epitope3d_dataset_20_test.csv', 
                        help='Tên file CSV test cho tập epitope3d')
    args = parser.parse_args()
    root = args.root
    device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
    
    os.system(f'cd {root} && mkdir PDB purePDB feat dssp graph')
    # model=None
    import getpass
    from esm.sdk import client

    # 1. Nhập token (Chỉ cần nhập 1 lần khi bắt đầu chạy script)
    token = '52GkN1RAZoG8IqpGSIuwMT'

    # 2. Khởi tạo Cloud Client (Thay vì Local Model)
    # Model "esmc-6b-2024-12" là bản 6B mới nhất trên Cloud
    print("[INFO] Đang kết nối tới Forge API cho ESM-C 6B...")
    model = client(
        model="esmc-6b-2024-12", 
        url="https://forge.evolutionaryscale.ai", 
        token=token
    )
    print("[INFO] Loading ESM-3...")
    
    esm3_model = client(
        model="esm3-medium-2024-08",
        url="https://forge.evolutionaryscale.ai",
        token=token
    )
    
    print("Model connected successfully!")
    
    print("[INFO] Đang xử lý tập dữ liệu Epitope3D (Đã chia sẵn Train/Test)...")

    print(f"--> Xử lý tập Train: {args.train_csv}")
    trainset = initial_epitope3D(args.train_csv, root, model, esm3_model, device)
        
    print(f"--> Xử lý tập Test: {args.test_csv}")
    testset = initial_epitope3D(args.test_csv, root, model, esm3_model, device)
        
    trainset = [i for i in trainset if len(i) < 1024 and getattr(i, 'label', None) is not None and i.label.sum() > 0]
    testset = [i for i in testset if len(i) < 1024 and getattr(i, 'label', None) is not None and i.label.sum() > 0]

    np.random.seed(42) 
    idx = np.random.permutation(len(trainset))
    with open(f'{root}/train.pkl','wb') as f:
        pk.dump(trainset, f)
    with open(f'{root}/test.pkl','wb') as f:
        pk.dump(testset, f)

    np.save(f'{root}/cross-validation.npy', idx)
    print(f"[INFO] TỔNG KẾT -> Train: {len(trainset)} chains, Test: {len(testset)} chains, CV idx shape: {idx.shape}")    
