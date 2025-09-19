import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Optional


class CrossSpaceDiscriminator(nn.Module):
    def __init__(self, dim_s: int, dim_t: int, hidden: int = 256, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = dim_s + dim_t
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
            if dropout > 0.0:
                layers += [nn.Dropout(p=dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_s, z_t], dim=1))  
    

class FactorVAEModule(L.LightningModule):

    def __init__(
        self,
        dim_s: int,
        dim_t: int,
        hidden: int = 256,
        depth: int = 2,
        dropout: float = 0.0,
        lr: float = 1e-3,
        wd: float = 0.0,
        betas=(0.9, 0.999),
        scheduler: Optional[str] = "cosine", 
        step_lr_gamma: float = 0.5,
        step_lr_milestones: tuple = (30, 60, 90),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.D = CrossSpaceDiscriminator(dim_s, dim_t, hidden, depth, dropout)


    def _bce_step(self, z_s: torch.Tensor, z_t: torch.Tensor):

        B = z_s.size(0)

        logits_joint = self.D(z_s, z_t)              
        y_joint = torch.ones_like(logits_joint)

        perm = torch.randperm(B, device=z_s.device)
        logits_marg = self.D(z_s, z_t[perm])       
        y_marg = torch.zeros_like(logits_marg)

        logits = torch.cat([logits_joint, logits_marg], dim=0)
        targets = torch.cat([y_joint, y_marg], dim=0)

        loss = F.binary_cross_entropy_with_logits(logits, targets)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            acc = (pred.eq(targets).float().mean())


            p_joint = torch.sigmoid(logits_joint)

            p_joint = torch.clamp(p_joint, 1e-6, 1 - 1e-6)
            tc_est = torch.mean(torch.log(p_joint) - torch.log(1.0 - p_joint))

        return loss, acc, tc_est

    def training_step(self, batch, _):
        z_s, z_t = batch
        loss, acc, tc_est = self._bce_step(z_s, z_t)
        self.log("train/loss_disc", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/tc_est", tc_est, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        z_s, z_t = batch
        loss, acc, tc_est = self._bce_step(z_s, z_t)
        self.log("val/loss_disc", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True)
        self.log("val/tc_est", tc_est, prog_bar=True, on_epoch=True)

    def test_step(self, batch, _):
        z_s, z_t = batch
        loss, acc, tc_est = self._bce_step(z_s, z_t)
        self.log("test/loss_disc", loss, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)
        self.log("test/tc_est", tc_est, on_epoch=True)


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=self.hparams.betas, weight_decay=self.hparams.wd)

        if self.hparams.scheduler is None:
            return opt

        if self.hparams.scheduler == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs if self.trainer else 100)
        elif self.hparams.scheduler == "step":
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(self.hparams.step_lr_milestones), gamma=self.hparams.step_lr_gamma)
        else:
            return opt

        return {"optimizer": opt, "lr_scheduler": sch}