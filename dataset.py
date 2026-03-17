#dataset.py
import os
import esm
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
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        seq=self.data[idx]
        feat=torch.cat([seq.feat,seq.dssp],1)
        return {
            'feat':feat,
            'label':seq.label,
            'edge_index': seq.edge_index,
            'edge_attr': seq.edge_attr,
            'pos': seq.pos
        }
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/Epitope3D', help='dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    args = parser.parse_args()
    root = args.root
    device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
    
    os.system(f'cd {root} && mkdir PDB purePDB feat dssp graph')
    # model=None
    model,_=esm.pretrained.esm2_t36_3B_UR50D()
    model=model.to(device)
    model.eval()
    print("[INFO] Đang xử lý tập dữ liệu Epitope3D (Đã chia sẵn Train/Test)...")

    print(f"--> Xử lý tập Train: {args.train_csv}")
    trainset = initial_epitope3D(args.train_csv, root, model, device)
        
    print(f"--> Xử lý tập Test: {args.test_csv}")
    testset = initial_epitope3D(args.test_csv, root, model, device)
        
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

#     train='total.csv'
#     initial(train,root,model,device)
#     with open(f'{root}/total.pkl','rb') as f:
#         dataset=pk.load(f)
#     dates={i.name:i.date for i in dataset}
# #     with open(f'{root}/date.pkl','rb') as f:
# #         dates=pk.load(f)
#     filt_data=[]
#     for i in dataset:
#         if len(i)<1024 and i.label.sum()>0:
#             filt_data.append(i)
#     month={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
#     trainset,valset,testset=[],[],[]
#     D,M,Y=[],[],[]
#     test=20210401
#     dates_=[]
#     for i in filt_data:
#         d,m,y=dates[i.name]
#         d,m,y=int(d),month[m],int(y)
#         if y<23:
#             y+=2000
#         else:
#             y+=1900
#         date=y*10000+m*100+d
#         if date<test:
#             dates_.append(date)
#             trainset.append(i)
#         else:
#             testset.append(i)
#     with open(f'{root}/train.pkl','wb') as f:
#         pk.dump(trainset,f)
#     with open(f'{root}/test.pkl','wb') as f:
#         pk.dump(testset,f)
#     idx=np.array(dates_).argsort()
#     np.save(f'{root}/cross-validation.npy',idx)
