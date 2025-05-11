import torch
import torch.nn.functional as F



def kl(mu: torch.Tensor, logvar: torch.Tensor, by_latent: bool=False) -> torch.Tensor:
    if by_latent :
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    else:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1,).mean()


def mse(x_hat_logits: torch.Tensor, x: torch.Tensor) :
    batch_size = x.size(0)

    return F.cross_entropy(x_hat_logits, x, reduction="sum") / batch_size