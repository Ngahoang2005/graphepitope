import os
import torch
import random
import warnings
import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tool import METRICS
from model import GraphBepi
from dataset import PDB, collate_fn

warnings.simplefilter('ignore')

def seed_everything(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--dataset', type=str, default='BCE_633')
parser.add_argument('--logger', type=str, default='./log')
parser.add_argument('--tag', type=str, default='GraphBepi_HGNN')
parser.add_argument('--root', type=str, default='')
parser.add_argument('--hcl', type=float, default=0.1)
args = parser.parse_args()

seed_everything(args.seed)
root = args.root

trainset = PDB(mode='train', fold=args.fold, root=root)
valset = PDB(mode='val', fold=args.fold, root=root)
testset = PDB(mode='test', fold=args.fold, root=root)

if args.fold == -1:
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=4)
else:
    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=4)

test_loader = DataLoader(testset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=4)

log_name = f'{args.dataset}_{args.tag}'
metrics = METRICS(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')

model = GraphBepi(
    feat_dim=1280,
    exfeat_dim=1280,
    hidden_dim=args.hidden,
    hcl_weight=args.hcl,
    augment_eps=0.05,
    dropout=0.2,
    lr=args.lr,
    metrics=metrics,
    result_path=f'./model/{log_name}'
)

es = EarlyStopping('val_AUPRC', patience=40, mode='max')
mc = ModelCheckpoint(
    dirpath=f'./model/{log_name}/',
    filename=f'model_{args.fold}',
    monitor='val_AUPRC',
    mode='max',
    save_weights_only=True
)

logger = TensorBoardLogger(args.logger, name=f"{log_name}_{args.fold}")

trainer = pl.Trainer(
    accelerator="cpu" if args.gpu == -1 else "gpu",
    devices=1 if args.gpu != -1 else None,
    max_epochs=args.epochs,
    callbacks=[mc, es],
    logger=logger,
    check_val_every_n_epoch=1,
    gradient_clip_val=1.0
)

ckpt_path = f'./model/{log_name}/model_{args.fold}.ckpt'
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

trainer.fit(model, train_loader, val_loader)

model.load_state_dict(torch.load(mc.best_model_path)['state_dict'])

trainer.test(model, test_loader)

if os.path.exists(f'./model/{log_name}/result.pkl'):
    os.rename(f'./model/{log_name}/result.pkl', f'./model/{log_name}/result_{args.fold}.pkl')