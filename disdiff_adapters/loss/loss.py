import torch
import torch.nn.functional as F
import torch.nn as nn

def kl(mu: torch.Tensor, logvar: torch.Tensor, by_latent: bool=False) -> torch.Tensor:
    
    if by_latent:
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
    else:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def mse(x_hat_logits: torch.Tensor, x: torch.Tensor) :
    return F.mse_loss(x_hat_logits, x, reduction="mean")

def cross_cov(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=0)    
    b = b - b.mean(dim=0)

    cov = (a.T @ b) / (a.size(0)-1)       
    return cov/(a.std()*b.std())     

def decorrelate_params(mu_s, logvar_s, mu_t, logvar_t,):
    return torch.norm(cross_cov(mu_s, mu_t), p="fro")

    
#InfoNCE supervised
class InfoNCESupervised(nn.Module) :
    def __init__(self, temperature: float=0.07, eps: float=1e-8) :
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor,
                    labels: torch.Tensor,) -> torch.Tensor:
  
        device = z.device
        batch_size = z.size(0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.t()) / self.temperature

        mask_self = torch.eye(batch_size, device=device).bool()
        sim.masked_fill_(mask_self, -1e9)

        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()) & ~mask_self  # (B, B)


        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1) 
        numer = (exp_sim * mask_pos.float()).sum(dim=1)

        loss = -torch.log((numer + self.eps) / (denom + self.eps))

        return loss.mean()