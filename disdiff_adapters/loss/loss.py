import torch
import torch.nn.functional as F

def kl(mu: torch.Tensor, logvar: torch.Tensor, by_latent: bool=False) -> torch.Tensor:
    
    if by_latent:
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
    else:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def mse(x_hat_logits: torch.Tensor, x: torch.Tensor) :
    return F.mse_loss(x_hat_logits, x, reduction="mean")

def _cross_frobenius(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Frobenius² de la matrice de corrélation batch × dim.  a,b : [B, d]
    """
    a = a - a.mean(dim=0)           # centrage
    b = b - b.mean(dim=0)
    a = F.normalize(a, dim=0)       # variance ≃ 1
    b = F.normalize(b, dim=0)

    c = (a.T @ b) / a.size(0)       # [d, d]  coefficients de Pearson
    return (c ** 2).sum()           # ‖C‖_F²

def decorrelate_params(mu_s, logvar_s, mu_t, logvar_t,
                       w_mu=1.0, w_var=1.0):
    """
    Loss = w_mu  * ‖corr(μ_s, μ_t)‖²_F +
           w_var * ‖corr(logσ²_s, logσ²_t)‖²_F
    """
    loss_mu  = _cross_frobenius(mu_s,     mu_t)
    loss_var = _cross_frobenius(logvar_s, logvar_t)
    return w_mu * loss_mu + w_var * loss_var