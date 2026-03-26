import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from HGNN import HierarchicalGNN

class GraphBepi(pl.LightningModule):
    def __init__(
        self, 
        feat_dim=1280, 
        exfeat_dim=1280, 
        hidden_dim=256, 
        augment_eps=0.05, 
        dropout=0.2, 
        lr=1e-4, 
        hcl_weight=0.1,
        metrics=None, 
        result_path=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.metrics = metrics
        self.path = result_path
        self.lr = lr
        self.hcl_weight = hcl_weight

        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []
        self.loss_fn = nn.BCELoss()

        self.W_v = nn.Linear(feat_dim, hidden_dim)
        self.W_u1 = nn.Linear(exfeat_dim, hidden_dim)

        self.hgnn = HierarchicalGNN(hidden_dim * 2, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_hcl_loss(self, layer_outputs, labels):
        hcl_loss = 0
        labels = labels.float()
        for z in layer_outputs:
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_mean = z[pos_mask].mean(0)
                neg_mean = z[neg_mask].mean(0)
                hcl_loss += F.mse_loss(pos_mean, neg_mean) * -1.0
        return hcl_loss

    def forward(self, x, pos, batch_idx=None):
        esm_feat = x[:, :-self.hparams.exfeat_dim]
        saprot_feat = x[:, -self.hparams.exfeat_dim:]

        v = F.elu(self.W_v(esm_feat))
        u = F.elu(self.W_u1(saprot_feat))
        
        combined_x = torch.cat([v, u], dim=-1)

        if self.training and self.hparams.augment_eps > 0:
            combined_x = combined_x + torch.randn_like(combined_x) * self.hparams.augment_eps

        layer_feats, final_feat = self.hgnn(combined_x, pos, batch_idx)
        out = self.classifier(final_feat)
        return out, layer_feats

    def training_step(self, batch, batch_idx):
        x, pos, y, batch_ptr = batch.x, batch.pos, batch.y, batch.batch
        pred, layer_feats = self(x, pos, batch_ptr)
        pred = pred.squeeze(-1)

        loss_bce = self.loss_fn(pred, y.float())
        loss_hcl = self.compute_hcl_loss(layer_feats, y)
        total_loss = loss_bce + self.hcl_weight * loss_hcl

        self.log('train_loss', total_loss, on_epoch=True, prog_bar=True)
        if self.metrics is not None:
            result = self.metrics.calc_prc(pred.detach(), y.detach())
            self.log('train_auc', result['AUROC'], on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, pos, y, batch_ptr = batch.x, batch.pos, batch.y, batch.batch
        pred, _ = self(x, pos, batch_ptr)
        pred = pred.squeeze(-1)
        self.val_preds.append(pred.detach())
        self.val_labels.append(y.detach())
        return self.loss_fn(pred, y.float())

    def on_validation_epoch_end(self):
        if not self.val_preds: return
        pred = torch.cat(self.val_preds, 0)
        y = torch.cat(self.val_labels, 0)
        self.val_preds.clear(); self.val_labels.clear()

        loss = self.loss_fn(pred, y.float())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        if self.metrics is not None:
            result = self.metrics(pred, y)
            for k, v in result.items():
                self.log(f'val_{k}', v, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, pos, y, batch_ptr = batch.x, batch.pos, batch.y, batch.batch
        pred, _ = self(x, pos, batch_ptr)
        self.test_preds.append(pred.detach().squeeze(-1))
        self.test_labels.append(y.detach())

    def on_test_epoch_end(self):
        if not self.test_preds: return
        pred = torch.cat(self.test_preds, 0)
        y = torch.cat(self.test_labels, 0)
        self.test_preds.clear(); self.test_labels.clear()

        if self.path:
            os.makedirs(self.path, exist_ok=True)
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')

        if self.metrics is not None:
            result = self.metrics(pred, y)
            for k, v in result.items():
                self.log(f'test_{k}', v, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)