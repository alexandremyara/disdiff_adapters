import torch
from lightning import LightningModule
import matplotlib.pyplot as plt

from disdiff_adaptaters.arch.vae import *
from disdiff_adaptaters.utils import sample_from, pca_latent, display
from disdiff_adaptaters.loss import *

class _VAE(torch.nn.Module) :
    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim: int) :
        
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels,
                               img_size=img_size,
                               latent_dim=latent_dim,)
        
        self.decoder = Decoder(out_channels=in_channels,
                               img_size=img_size,
                               latent_dim=latent_dim,
                               out_encoder_shape=self.encoder.out_encoder_shape)
        
    def forward(self, images: torch.Tensor, test: bool=False) :
        mus_logvars = self.encoder(images)
        z = sample_from(mus_logvars, test)
        image_hat_logits = self.decoder(z)

        return image_hat_logits, mus_logvars

class VAEModule(LightningModule) :

    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim: int,
                 beta: float=1.0,
                 warm_up: bool=False) :
        
        super().__init__()
        self.save_hyperparameters()

        self.model = _VAE(in_channels=self.hparams.in_channels,
                          img_size=self.hparams.img_size,
                          latent_dim=self.hparams.latent_dim)
        self.images_buff = None
    
            
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-2)
    
    def generate(self, nb_samples: int=8) -> torch.Tensor :
        eps = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim])).to(self.device, torch.float32)

        x_hat_logits = self.model.decoder(eps)
        return x_hat_logits
    
    def show_reconstruct(self, images: torch.Tensor) :
        images = images.to(self.device)
        images_gen, _ = self(images, test=True)

        fig, axes = plt.subplots(len(images), 2, figsize=(7, 20))

        for i in range(len(images)) :
            images_proc = (images[i]*255).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()
            images_gen_proc = (images_gen[i]*255).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()

            axes[i,0].imshow(images_proc)
            axes[i,1].imshow(images_gen_proc)

            axes[i,0].set_title("original")
            axes[i,1].set_title("reco")
        plt.show()
    
    def forward(self, images: torch.Tensor, test: bool=False) -> tuple[torch.Tensor]:
        image_hat_logits, mus_logvars= self.model(images, test)

        return image_hat_logits, mus_logvars
    
    def loss(self, image_hat_logits, mus_logvars, images, log_components=False) -> float :
        max_beta = self.hparams.beta
        mus, logvars = mus_logvars

        # beta warm-up
        if self.hparams.warm_up :
            start_epoch = int(self.trainer.max_epochs*1/5)
            epoch_limit = int(self.trainer.max_epochs*2/5)

            if self.current_epoch < start_epoch:
                beta = 0.0
            elif self.current_epoch <= epoch_limit:
                progress = (self.current_epoch - start_epoch) / (epoch_limit - start_epoch)
                beta = max_beta * progress
            else:
                beta = max_beta
        else : beta=max_beta
        
        weighted_kl = beta * kl(mus, logvars)

        reco = mse(image_hat_logits, images)

        if log_components :
            self.log("loss/kl", weighted_kl)
            self.log("loss/reco", reco)
            self.log("loss/beta", beta)

        return weighted_kl+reco
    
    def training_step(self, batch: tuple[torch.Tensor]) -> float:
        images, labels = batch
        image_hat_logits,mus_logvars = self.forward(images)
        loss = self.loss(image_hat_logits, mus_logvars, images, log_components=True)

        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor]):
        images, labels = batch
        image_hat_logits, mus_logvars = self.forward(images)
        loss = self.loss(image_hat_logits, mus_logvars, images)

        self.log("loss/val", loss)
        print(f"Val loss: {loss}")
    
    def test_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        image_hat_logits, mus_logvars = self.forward(images)

        weighted_kl= self.hparams.beta*kl(*mus_logvars)
        reco = mse(image_hat_logits, images)

        self.log("loss/reco_test", reco)
        self.log("loss/kl_test", weighted_kl)
        self.log("loss/test", reco+weighted_kl)

        if self.images_buff is None : self.images_buff = images


    def on_test_end(self):
        images_gen = self.generate()
        labels_gen = torch.zeros([images_gen.shape[0],1])

        display((images_gen.detach().to("cpu"), labels_gen.detach().to("cpu")))

        self.logger.experiment.add_figure("img/gen", plt.gcf())

        self.show_reconstruct(self.images_buff)
        self.logger.experiment.add_figure("img/reco", plt.gcf())        




