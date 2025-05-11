#Import from site packages
import lightning as L
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

#Import from this module (vae)
from disdiff_adaptaters.arch.vae.encoder import Encoder
from disdiff_adaptaters.arch.vae.decoder import Decoder

#Import from other modules
from disdiff_adaptaters.loss import kl, mse
from disdiff_adaptaters.utils import sample_from

class VAEModule(L.LightningModule):


    def __init__(
        self,
        in_channels: int=4,
        img_size: int=64,
        latent_dim: int=2,
        beta: float=1.0):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.encoder = Encoder(in_channels=self.hparams.in_channels,
                               img_size=self.hparams.img_size,
                               latent_dim=self.hparams.latent_dim)
        
        self.decoder = Decoder(out_channels=self.hparams.in_channels,
                               img_size=self.hparams.img_size,
                               latent_dim=self.hparams.latent_dim,
                               out_encoder_shape=self.encoder.out_encoder_shape)       

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=6e-5, weight_decay=1e-2)
          
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
        recon_loss = mse(x_hat_logits, x)
        kl_div = kl(mu, logvar)

        weighted_kl_div = self.hparams.beta * kl_div
        
        if log_components:
            kl_by_latent = kl(mu, logvar, by_latent=True)
            
            self.log("loss/recon", recon_loss)
            self.log("loss/kl_div", weighted_kl_div)

            for i, marginal_kl in enumerate(kl_by_latent):
                self.log(f"loss/marginal_kl_div/dim_{i}", marginal_kl)
        
        return recon_loss + weighted_kl_div

    def forward(
        self,
        x: torch.Tensor,
        test: bool=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE encoder and decoder.
        
        Args:
            x (torch.Tensor): image
        
        Returns:
            mu (torch.Tensor): Mean of approximate posterior.
            logvar (torch.Tensor): Log-variance of approximate posterior.
            z (torch.Tensor): Latent representation of input.
            x_hat_logits (torch.Tensor): Logits of reconstruction of input.
        """
        mu, logvar = self.encoder(x)
        z = sample_from((mu, logvar), test)
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
        pass
