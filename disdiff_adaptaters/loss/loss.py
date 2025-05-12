import torch
import torch.nn.functional as F



def kl(mu: torch.Tensor, logvar: torch.Tensor, by_latent: bool=False) -> torch.Tensor:
        if by_latent:
            return -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=0,
            )
        
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)



def mse(x_hat_logits: torch.Tensor, x: torch.Tensor) :
    return F.mse_loss(x_hat_logits, x, reduction="mean")