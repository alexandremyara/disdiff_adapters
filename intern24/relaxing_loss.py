import torch
import torch.nn as nn

class RelaxingContrastiveLoss(nn.Module):
    def __init__(
        self,
        coef: float = 1.0,
        temperature: float = 1.0,
        margin: float = 0.1,
        squared: bool = False,
        activation_epoch_ratio: float = 0.5,  # Activate after this ratio of total epochs
    ):
        super().__init__()
        self.coef = coef
        self.temperature = temperature
        self.margin = margin
        self.squared = squared
        self.loss_fn = self.squared_loss if squared else self.non_squared_loss
        self.activation_epoch_ratio = activation_epoch_ratio
        self.current_epoch = 0
        self.total_epochs = None
        self.is_active = False

    def squared_loss(self, sim_matrix, mask):
        return torch.sum(
            mask * torch.relu(torch.square(sim_matrix) - self.margin * self.margin)
        ) / torch.sum(mask)

    def non_squared_loss(self, sim_matrix, mask):
        return torch.sum(mask * torch.relu(sim_matrix - self.margin)) / torch.sum(mask)

    def set_total_epochs(self, total_epochs: int):
        """Set total number of epochs at the start of training"""
        self.total_epochs = total_epochs

    def update_epoch(self, epoch: int):
        """Update current epoch number"""
        self.current_epoch = epoch
        if self.total_epochs is not None:
            self.is_active = (
                self.current_epoch / self.total_epochs
            ) >= self.activation_epoch_ratio

    def forward(self, z, y, z_f_mu, z_nf_mu, z_f, z_nf, z_hat):
        # If not active yet or total_epochs not set, return zero loss
        if not self.is_active or self.total_epochs is None:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Compute relaxing contrastive loss
        embeddings = torch.nn.functional.normalize(z_nf, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        mask = (y.unsqueeze(0) == y.unsqueeze(1)).float() - torch.eye(
            y.size(0), device=y.device
        )

        relaxing_loss = self.loss_fn(sim_matrix, mask)
        return self.coef * relaxing_loss