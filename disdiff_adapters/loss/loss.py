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
        embeddings = torch.nn.functional.normalize(z_f, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        negative_loss = self.negative_loss(sim_matrix, 1 - mask)

        mask = mask - torch.eye(y.size(0), device=y.device)
        positive_samples = mask * (1 - sim_matrix)
        positive_loss = torch.sum(positive_samples) / torch.sum(mask)

        return self.coef * (positive_loss + negative_loss)
    

class InfoNCELoss(nn.Module):
    def __init__(
        self,
        coef: float = 1.0,
        temperature: float = 0.1,
        projection_dim: int = 128,
        device="cuda",
    ):
        super().__init__()
        self.coef = coef
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.device = device

        self.project_f = nn.Sequential(
            nn.Linear(2, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        ).to(device)

        self.project_nf = nn.Sequential(
            nn.Linear(16, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        ).to(device)

    def forward(self, z, y, z_f_mu, z_nf_mu, z_f, z_nf, z_hat):
        batch_size = z_f.size(0)

        # Project to same dimension
        z_f_proj = self.project_f(z_f)
        z_nf_proj = self.project_nf(z_nf)

        # Normalize embeddings
        z_f_norm = F.normalize(z_f_proj, dim=1)
        z_nf_norm = F.normalize(z_nf_proj, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z_f_norm, z_nf_norm.T) / self.temperature

        # InfoNCE loss - we want to MINIMIZE MI, so we take negative of standard InfoNCE
        labels = torch.arange(batch_size, device=z_f.device)
        # Notice we swap the order here to turn minimization into maximization
        loss = -F.cross_entropy(sim_matrix, labels)  # THIS IS THE KEY CHANGE

        return self.coef * loss.abs()


###     Anti constrastive
def non_squared_loss(sim_matrix, mask, margin=0.1):
    return torch.sum(mask * torch.relu(sim_matrix - margin)) / torch.sum(mask)

def anti_info_nce(z, labels, temp=0.1, margin=0.1):

    # Compute relaxing contrastive loss
    embeddings = torch.nn.functional.normalize(z, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temp
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() - torch.eye(
        labels.size(0), device=labels.device
    )

    relaxing_loss = non_squared_loss(sim_matrix, mask, margin)
    return relaxing_loss