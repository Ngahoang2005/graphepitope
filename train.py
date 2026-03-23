#train.py
import os
import torch
import random
import warnings
import argparse
import numpy as np
import pickle as pk
import pytorch_lightning as pl
from tool import METRICS
from model import GraphBepi
from dataset import PDB,collate_fn,chain
from torch.utils.data import DataLoader,Dataset,random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback,EarlyStopping,ModelCheckpoint
warnings.simplefilter('ignore')
def seed_everything(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate.')
parser.add_argument('--gpu', type=int, default=0, help='gpu.')
parser.add_argument('--fold', type=int, default=1, help='dataset fold. set it -1 to use the whole trainset')
parser.add_argument('--seed', type=int, default=2022, help='random seed.')
parser.add_argument('--batch', type=int, default=4, help='batch size.')
parser.add_argument('--hidden', type=int, default=256, help='hidden dim.')
parser.add_argument('--epochs', type=int, default=300, help='max number of epochs.')
parser.add_argument('--dataset', type=str, default='BCE_633', help='dataset name.')
parser.add_argument('--logger', type=str, default='./log', help='logger path.')
parser.add_argument('--tag', type=str, default='GraphBepi', help='logger name.')
parser.add_argument('--root', type=str, default='', help='root path.')
args = parser.parse_args()

device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
seed_everything(args.seed)
root=args.root

trainset=PDB(mode='train',fold=args.fold,root=root)
valset=PDB(mode='val',fold=args.fold,root=root)
testset=PDB(mode='test',fold=args.fold,root=root)

if args.fold == -1:
    # Split trainset into train and val (80% train, 20% val)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
else:
    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

test_loader=DataLoader(testset,batch_size=args.batch,shuffle=False,collate_fn=collate_fn)
print("Train size:", len(trainset))
print("Val size:", len(valset))
print("Test size:", len(testset))
print("Train loader batches:", len(train_loader))
log_name=f'{args.dataset}_{args.tag}'
metrics=METRICS(device)
model=GraphBepi(
    feat_dim=2560,                     # esm2 representation dim
    hidden_dim=args.hidden,            # hidden representation dim
    exfeat_dim=13,                     # dssp feature dim
    edge_dim=51,                       # edge feature dim
    augment_eps=0.05,                  # random noise rate
    dropout=0.2,
    lr=args.lr,                        # learning rate
    metrics=metrics,                   # an implement to compute performance
    result_path=f'./model/{log_name}', # path to save temporary result file of testset
)

es=EarlyStopping('val_AUPRC',patience=40,mode='max')

mc=ModelCheckpoint(
    f'./model/{log_name}/',f'model_{args.fold}',
    'val_AUPRC',
    mode='max',
    save_weights_only=True, 
)
logger = TensorBoardLogger(
    args.logger, 
    name=log_name+f'_{args.fold}'
)
cb=[mc,es]
trainer = pl.Trainer(
    accelerator="cpu" if args.gpu==-1 else "gpu",
    devices=1 if args.gpu!=-1 else None,
    max_epochs=args.epochs,
    callbacks=cb,
    logger=logger,
    check_val_every_n_epoch=1,
    enable_progress_bar=False,  # Tắt progress bar để chỉ in log epoch
)

if os.path.exists(f'./model/{log_name}/model_{args.fold}.ckpt'):
    os.remove(f'./model/{log_name}/model_{args.fold}.ckpt')
trainer.fit(model, train_loader, val_loader)
model.load_state_dict(
    torch.load(f'./model/{log_name}/model_{args.fold}.ckpt')['state_dict'],
)
trainer = pl.Trainer(
    accelerator="cpu" if args.gpu==-1 else "gpu",
    devices=1 if args.gpu!=-1 else None,
    logger=None
)
result = trainer.test(model,test_loader)
os.rename(f'./model/{log_name}/result.pkl',f'./model/{log_name}/result_{args.fold}.pkl')