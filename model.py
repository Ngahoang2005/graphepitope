# model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from EGNN import EGNN   

import torch.nn.functional as F

class PULoss(nn.Module):
    def __init__(self, prior=0.9):
        super().__init__()
        self.prior = prior

    def forward(self, logits, target):
        y = target.float()
        pred = torch.sigmoid(logits)

        pos = (y == 1)
        unl = (y == 0)

        if pos.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        loss_pos = -torch.log(pred[pos] + 1e-8).mean()

        if unl.sum() == 0:
            return loss_pos

        loss_unl = -torch.log(1 - pred[unl] + 1e-8).mean()

        return self.prior * loss_pos + loss_unl
class GraphBepi(pl.LightningModule):
    def __init__(
        self, 
        feat_dim=2560, 
        hidden_dim=256, 
        exfeat_dim=13, 
        edge_dim=51, 
        augment_eps=0.05, 
        dropout=0.2, 
        lr=1e-4,            # <-- ĐÃ TĂNG LR ĐỂ MODEL HỌC ĐƯỢC
        num_egnn_layers=4,  # <-- CẤU HÌNH SỐ LỚP EGNN (Xếp chồng)
        metrics=None, 
        result_path=None
    ):
        super().__init__()
        self.metrics = metrics
        self.path = result_path
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []

        # ================== LOSS & HYPERPARAMS ==================
        self.loss_fn = PULoss(prior=0.9)  
        self.exfeat_dim = exfeat_dim
        self.augment_eps = augment_eps
        self.lr = lr

        # ================== LAYERS ==================
        self.W_v = nn.Linear(feat_dim, hidden_dim)          # ESM-2 projection
        self.W_u = nn.Linear(exfeat_dim, hidden_dim)        # DSSP projection
        
        # Đặc trưng sau khi nối ESM-2 và DSSP sẽ có số chiều là 2 * hidden_dim
        in_dim = 2 * hidden_dim
        
        # Khai báo ModuleList để xếp chồng (stacking) nhiều lớp EGNN
        self.egnn_layers = nn.ModuleList([
            EGNN(in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout, residual=True)
            for _ in range(num_egnn_layers)
        ])

        # ================== OUTPUT HEAD ==================
        # Sử dụng GELU thay vì ReLU giúp gradient mượt hơn với các mạng sâu
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            #nn.Sigmoid()
        )

        # Khởi tạo trọng số (Initialization)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ===================== FORWARD =====================
    def forward(self, feats, edge_attrs, edge_indices, poss):
        """
        feats: list of [L_i x (2560+13)]
        edge_attrs, edge_indices, poss: tương ứng cho từng protein
        """
        outs = []
        for V, edge_attr, edge_index, pos in zip(feats, edge_attrs, edge_indices, poss):
            V = V.float()
            
            # Project features
            esm_proj = self.W_v(V[:, :-self.exfeat_dim])
            dssp_proj = self.W_u(V[:, -self.exfeat_dim:])
            x = torch.cat([esm_proj, dssp_proj], dim=-1)

            # Chạy qua lần lượt từng lớp EGNN (Cập nhật cả node x và toạ độ pos)
            for egnn in self.egnn_layers:
                x, pos = egnn(x, edge_index, edge_attr, pos)
            
            outs.append(x)

        h = torch.cat(outs, dim=0)          # [total_residues, in_dim]
        return self.mlp(h).squeeze(-1)

    # ===================== EMBEDDING =====================
    def embed(self, feats, edge_attrs, edge_indices, poss):
        """Trả về embedding per-residue trước MLP (dùng để xuất cho XGBoost sau này)"""
        was_train = self.training
        self.eval()
        with torch.no_grad():
            outs = []
            for V, edge_attr, edge_index, pos in zip(feats, edge_attrs, edge_indices, poss):
                V = V.float()
                esm_proj = self.W_v(V[:, :-self.exfeat_dim])
                dssp_proj = self.W_u(V[:, -self.exfeat_dim:])
                x = torch.cat([esm_proj, dssp_proj], dim=-1)
                
                # Đi qua các lớp EGNN
                for egnn in self.egnn_layers:
                    x, pos = egnn(x, edge_index, edge_attr, pos)
                
                outs.append(x)
        
        if was_train:
            self.train()
        return outs

    # ===================== TRAINING / VALIDATION / TEST =====================
    def training_step(self, batch, batch_idx):
        feats, edge_attrs, edge_indices, poss, y = batch
        pred = self(feats, edge_attrs, edge_indices, poss)
        loss = self.loss_fn(pred, y.float())
        if batch_idx == 0:
            print("logits mean:", pred.mean().item(), "std:", pred.std().item())
        return loss

    def validation_step(self, batch, batch_idx):
        feats, edge_attrs, edge_indices, poss, y = batch
        pred = self(feats, edge_attrs, edge_indices, poss)
        self.val_preds.append(pred.detach())
        self.val_labels.append(y.detach())

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return
        pred = torch.cat(self.val_preds, 0)
        y = torch.cat(self.val_labels, 0)
        self.val_preds.clear()
        self.val_labels.clear()

        loss = self.loss_fn(pred, y.float())

        if self.metrics is not None:
            pred_prob = torch.sigmoid(pred)
            result = self.metrics(pred_prob, y)
            self.log('val_AUROC', result['AUROC'])
            self.log('val_AUPRC', result['AUPRC'])
            self.log('val_mcc',   result['MCC'])
            self.log('val_f1',    result['F1'])
            print(f"Epoch {self.current_epoch:03d}: val_loss={loss:.4f}, AUPRC={result['AUPRC']:.4f}, AUROC={result['AUROC']:.4f}")

    def test_step(self, batch, batch_idx):
        feats, edge_attrs, edge_indices, poss, y = batch
        pred = self(feats, edge_attrs, edge_indices, poss)
        self.test_preds.append(pred.detach())
        self.test_labels.append(y.detach())

    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        pred = torch.cat(self.test_preds, 0)
        y = torch.cat(self.test_labels, 0)
        self.test_preds.clear()
        self.test_labels.clear()

        loss = self.loss_fn(pred, y.float())

        if self.path:
            import os
            os.makedirs(self.path, exist_ok=True)
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')

        if self.metrics is not None:
            pred_prob = torch.sigmoid(pred)
            result = self.metrics(pred_prob, y)
            self.log('test_AUROC', result['AUROC'])
            self.log('test_AUPRC', result['AUPRC'])
            self.log('test_f1',    result['F1'])
            self.log('test_mcc',   result['MCC'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0
        }