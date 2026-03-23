# model.py (phiên bản hoàn chỉnh sau khi sửa)
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from EGNN import EGNN   

class PULoss(nn.Module):
    """PU Loss (Positive-Unlabeled) – xử lý imbalance epitope (~9-11%)"""
    def __init__(self, prior=0.1):          # prior = tỷ lệ positive trong dataset
        super().__init__()
        self.prior = prior

    def forward(self, pred, target):
        pos_mask = (target == 1)
        if pos_mask.any():
            loss_pos = -torch.log(pred[pos_mask] + 1e-8).mean()
        else:
            loss_pos = 0.0
        loss_unlab = -torch.log(1 - pred[~pos_mask] + 1e-8).mean()
        return self.prior * loss_pos + (1 - self.prior) * loss_unlab


class GraphBepi(pl.LightningModule):
    def __init__(
        self, 
        feat_dim=2560, 
        hidden_dim=256, 
        exfeat_dim=13, 
        edge_dim=51, 
        augment_eps=0.05, 
        dropout=0.2, 
        lr=1e-6, 
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
        self.loss_fn = PULoss(prior=0.1)          # ← PU Loss thay vì BCELoss
        self.exfeat_dim = exfeat_dim
        self.augment_eps = augment_eps
        self.lr = lr

        # ================== LAYERS (chỉ giữ graph branch) ==================
        self.W_v = nn.Linear(feat_dim, hidden_dim)          # ESM-2 projection
        self.W_u = nn.Linear(exfeat_dim, hidden_dim)        # DSSP projection
        self.egnn = EGNN(2 * hidden_dim, hidden_dim, edge_dim=51, dropout=dropout)

        # ================== OUTPUT HEAD ==================
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ===================== FORWARD (chỉ EGNN) =====================
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

            # Equivariant GNN + position (tọa độ Cα)
            x, _ = self.egnn(x, edge_index, edge_attr, pos)
            outs.append(x)

        h = torch.cat(outs, dim=0)          # [total_residues, hidden_dim]
        return self.mlp(h).squeeze(-1)

    # ===================== EMBEDDING =====================
    def embed(self, feats, edge_attrs, edge_indices, poss):
        """Trả về embedding per-residue trước MLP (dùng cho phân tích sau)"""
        was_train = self.training
        self.eval()
        with torch.no_grad():
            outs = []
            for V, edge_attr, edge_index, pos in zip(feats, edge_attrs, edge_indices, poss):
                V = V.float()
                esm_proj = self.W_v(V[:, :-self.exfeat_dim])
                dssp_proj = self.W_u(V[:, -self.exfeat_dim:])
                x = torch.cat([esm_proj, dssp_proj], dim=-1)
                x, _ = self.egnn(x, edge_index, edge_attr, pos)
                outs.append(x)
        if was_train:
            self.train()
        return outs

    # ===================== TRAINING / VALIDATION / TEST =====================
    def training_step(self, batch, batch_idx):
        feats, edge_attrs, edge_indices, poss, y = batch
        pred = self(feats, edge_attrs, edge_indices, poss)
        loss = self.loss_fn(pred, y.float())
        #self.log('train_loss', loss, prog_bar=True)
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
        #self.log('val_loss', loss, prog_bar=True)

        if self.metrics is not None:
            result = self.metrics(pred, y)
            self.log('val_AUROC', result['AUROC'])
            self.log('val_AUPRC', result['AUPRC'])
            self.log('val_mcc',   result['MCC'])
            self.log('val_f1',    result['F1'])
            print(f"Epoch {self.current_epoch}: val_loss={loss:.4f}, AUPRC={result['AUPRC']:.4f}, AUROC={result['AUROC']:.4f}")

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
            result = self.metrics(pred, y)
            self.log('test_AUROC', result['AUROC'])
            self.log('test_AUPRC', result['AUPRC'])
            self.log('test_f1',    result['F1'])
            self.log('test_mcc',   result['MCC'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=1e-5)
