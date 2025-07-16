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
    return cov     

def decorrelate_params(mu_s, logvar_s, mu_t, logvar_t,
                       l_mu=1.0, l_var=1.0):
    loss_mu  = torch.norm(cross_cov(mu_s, mu_t), p="fro")
    loss_var = torch.norm(cross_cov(logvar_s, logvar_t), p="fro")
    return l_mu * loss_mu + l_var * loss_var

class MultiClassSupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, y):
        z = F.normalize(z, dim=1)
        num_classes = y.max().item() + 1  
        protos = torch.stack([z[y == i].mean(0) for i in range(num_classes)], dim=0)  
        logits = z @ protos.T / self.temperature
        return F.cross_entropy(logits, y)
    
class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        coef: float = 1.0,
        temperature: float = 1.0,
        margin: float = 0.5,
        squared: bool = False,
    ):
        super().__init__()
        self.coef = coef
        self.temperature = temperature
        self.margin = margin

        self.negative_loss = (
            self.squared_negative_loss if squared else self.non_squared_negative_loss
        )

    def squared_negative_loss(self, sim_matrix, negative_mask):
        return torch.sum(
            negative_mask
            * torch.relu(torch.square(sim_matrix) - self.margin * self.margin)
        ) / torch.sum(negative_mask)

    def non_squared_negative_loss(self, sim_matrix, negative_mask):
        return torch.sum(
            negative_mask * torch.relu(sim_matrix - self.margin)
        ) / torch.sum(negative_mask)

    def forward(self, y, z_f,):
        ## y : [B,]
        ## z_f : [B, 2]
        embeddings = torch.nn.functional.normalize(z_f, p=2, dim=1)
        sim_matrix = torch.exp(torch.matmul(embeddings, embeddings.T) / self.temperature)

        mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        negative_loss = self.negative_loss(sim_matrix, 1 - mask)

        mask = mask - torch.eye(y.size(0), device=y.device)
        positive_samples = mask * (1 - sim_matrix) 
        print(sim_matrix.min())
        positive_loss = torch.sum(positive_samples) / torch.sum(mask) # < 0
        print(positive_loss, negative_loss)
        return self.coef * (positive_loss + negative_loss)
    
#InfoNCE supervisé
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