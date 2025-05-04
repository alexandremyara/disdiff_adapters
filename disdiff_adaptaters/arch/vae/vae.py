import lightning as L
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from disdiff_adaptaters.arch.vae.encoder import Encoder
from disdiff_adaptaters.arch.vae.decoder import Decoder


class VAE(L.LightningModule):


    def __init__(
        self,
        in_channels: int=4,
        latent_dim: int=2,
        loss_reg: str="beta_vae",
        beta: float=1.0,
        gamma: float=1.0,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.encoder = Encoder(self.hparams.in_channels, self.hparams.latent_dim)
        self.decoder = Decoder(self.hparams.in_channels, self.hparams.latent_dim)
        
        # To keep track of test set and generated samples during test time, to
        # compute FRDS
        self.x_buffer: list[torch.Tensor] = []
        self.x_fake_logits_buffer: list[torch.Tensor] = []

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=6e-5, weight_decay=1e-2)
    
    def _kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        marginal: bool=False,
    ) -> torch.Tensor:
        if marginal:
            return -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=0,
            )
        
        return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        ).mean()
        
    def reconstruction_loss(self, x: torch.Tensor, x_hat_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the reconstruction loss using cross-entropy.
        
        Args:
            x (torch.Tensor): One-hot encoded input segmentations.
            x_hat_logits (torch.Tensor): Logits of reconstruction of input
                (output of decoder).
        
        Returns:
            recon_loss (torch.Tensor): Reconstruction loss.
        """
        batch_size = x.size(0)
        return F.cross_entropy(x_hat_logits, x, reduction="sum") / batch_size
    
    def loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        x_hat_logits: torch.Tensor,
        log_components: bool=True,
    ) -> torch.Tensor:
        """
        Compute the beta-VAE loss: sum of reconstruction loss and beta-weighted
        KL divergence regulariser term.
        """
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        kl_div = self._kl_divergence(mu, logvar)

        weighted_kl_div = self.hparams.beta * kl_div
        
        if log_components:
            marginal_kl_div = self._kl_divergence(mu, logvar, marginal=True)
            
            self.log("loss/recon", recon_loss)
            self.log("loss/kl_div", weighted_kl_div)
            for i, marginal_kl in enumerate(marginal_kl_div):
                self.log(f"loss/marginal_kl_div/dim_{i}", marginal_kl)
        
        return recon_loss + weighted_kl_div
    
    def _reparameterise(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        test: bool=False,
    ) -> torch.Tensor:
        # z_m = mu(x_m) + sigma(x_m) * epsilon
        # epsilon ~ N(0, 1)
        
        eps = torch.randn_like(logvar)
        
        if test:
            return mu
        return mu + torch.exp(0.5 * logvar) * eps
    
    def get_latent(self, x: torch.Tensor, test: bool=False) -> torch.Tensor:
        """
        Given an input tensor, return its latent representation z by passing it
        through the encoder.
        """
        mu, logvar = self.encoder(2 * x - 1.0)
        return self._reparameterise(mu, logvar, test)

    def forward(
        self,
        x: torch.Tensor,
        test: bool=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE encoder and decoder.
        
        Args:
            x (torch.Tensor): One-hot encoded input segmentations.
        
        Returns:
            mu (torch.Tensor): Mean of approximate posterior.
            logvar (torch.Tensor): Log-variance of approximate posterior.
            z (torch.Tensor): Latent representation of input.
            x_hat_logits (torch.Tensor): Logits of reconstruction of input.
        """
        mu, logvar = self.encoder(2 * x - 1.0)
        z = self._reparameterise(mu, logvar, test)
        x_hat_logits = self.decoder(z)
        return mu, logvar, z, x_hat_logits
    
    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar, z, x_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(x, mu, logvar, z, x_hat_logits)
        self.log("loss/train", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss
    
    def validation_step(self, x: torch.Tensor):
        mu, logvar, z, x_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(x, mu, logvar, z, x_hat_logits, log_components=False)
        self.log("loss/val", loss)
        
        print(f"Val loss: {loss}")
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Testing uses ACDC3DDataModule instead of ACDCDataModule to compute 3D
        Dice scores.
        """
        _, x, condition, ed = batch
        
        condition_label = f"condition_{int(condition)}"
        phase_label = "ed" if ed else "es"
        
        # 3D data module ensures 1 batch only, but each data point is 4D of
        # shape (S, C, W, H) where S is the number of slices.
        x = x.squeeze(0)
        
        self.log_reconstruction_metrics(x, condition_label, phase_label)
        self.log_generation_metrics(x)
        
        self.x_buffer.append(x)
    
