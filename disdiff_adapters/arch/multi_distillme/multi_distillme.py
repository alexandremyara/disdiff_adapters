import torch
import torch.nn as nn
from lightning import LightningModule
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils

from disdiff_adapters.arch.vae import *
from disdiff_adapters.utils import sample_from, pca_latent, display
from disdiff_adapters.loss import *


class _MultiDistillMe(torch.nn.Module) : 
    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim_s: int,
                 latent_dim_t: int,
                 res_block: nn.Module=ResidualBlock) :
        
        super().__init__()

        self.encoder_s = Encoder(in_channels=in_channels, 
                                 img_size=img_size,
                                 latent_dim=latent_dim_s,
                                 res_block=res_block)
        
        self.encoder_t = Encoder(in_channels=in_channels, 
                                 img_size=img_size,
                                 latent_dim=latent_dim_t,
                                 res_block=res_block)
        
        self.merge_operation = lambda z_s, z_t : torch.cat([z_s, z_t], dim=1)

        self.decoder = Decoder(out_channels=in_channels,
                               img_size=img_size,
                               latent_dim=latent_dim_s+latent_dim_t,
                               res_block=res_block,
                               out_encoder_shape=self.encoder_s.out_encoder_shape)
        
        self.labels_buff = []
        self.latent_buff = []
        
    def forward(self, images: torch.Tensor) :

        #forward s - semble encoder la couleur
        mus_logvars_s = self.encoder_s(images)
        z_s = sample_from(mus_logvars_s)

        #forward_t - semble encoder la forme
        mus_logvars_t = self.encoder_t(images)
        z_t = sample_from(mus_logvars_t)

        #merge latent vector from s and t
        z = self.merge_operation(z_s, z_t)

        #decoder
        image_hat_logits = self.decoder(z)

        return mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z
    
class MultiDistillMeModule(LightningModule) :

    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim_s: int,
                 latent_dim_t: int,
                 res_block: nn.Module=ResidualBlock,
                 beta_s: float=1.0,
                 beta_t: float=1.0,
                 warm_up:bool =False,
                 kl_weight: float= 10e-4,
                 l_cov: float=1,) :
        
        super().__init__()
        self.save_hyperparameters()

        self.model = _MultiDistillMe(in_channels=self.hparams.in_channels,
                                     img_size=self.hparams.img_size,
                                     latent_dim_s=self.hparams.latent_dim_s,
                                     latent_dim_t=self.hparams.latent_dim_t,
                                     res_block=self.hparams.res_block)
            

        self.images_test_buff = None
        self.images_train_buff = None
        self.z_ref = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())
    
    def generate(self, nb_samples: int=8) :
        eps_s = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_s])).to(self.device, torch.float32)
        eps_t = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_t])).to(self.device, torch.float32)


        z = self.model.merge_operation(eps_s, eps_t)

        x_hat_logits = self.model.decoder(z)
        return x_hat_logits

    def generate_cond(self, nb_samples: int=8, cond: str="t") :
        if cond == "t" :
            eps_t = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_t])).to(self.device, torch.float32)
            z_s = torch.stack(nb_samples*[self.z_ref["s"]])
            z = self.model.merge_operation(z_s, eps_t)
        elif cond == "s" :
            eps_s = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_s])).to(self.device, torch.float32)
            z_t = torch.stack(nb_samples*[self.z_ref["t"]])
            z = self.model.merge_operation(eps_s, z_t)
        else : raise ValueError("cond has to be equal to ")

        x_hat_logits = self.model.decoder(z)
        return x_hat_logits

    def show_reconstruct(self, images: torch.Tensor) :
        images = images.to(self.device)[:8]
        mus_logvars_s, mus_logvars_t, images_gen, z_s, z_t, z = self(images, test=True)

        fig, axes = plt.subplots(len(images), 2, figsize=(7, 20))

        for i in range(len(images)) :
            img = images[i]
            img_gen = images_gen[i]

            images_proc = (255*((img - img.min()) / (img.max() - img.min() + 1e-8))).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()
            images_gen_proc = (255*((img_gen - img_gen.min()) / (img_gen.max() - img_gen.min() + 1e-8))).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()

            axes[i,0].imshow(images_proc)
            axes[i,1].imshow(images_gen_proc)

            axes[i,0].set_title("original")
            axes[i,1].set_title("reco")
        plt.tight_layout()
        plt.show()

    def forward(self, images: torch.Tensor, test=False) :
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.model(images)
        return mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z
    
    def loss(self, mus_logvars_s, mus_logvars_t, image_hat_logits, images, log_components=False) :

        weighted_kl_s = self.hparams.kl_weight*self.hparams.beta_s*kl(*mus_logvars_s)
        weighted_kl_t = self.hparams.kl_weight*self.hparams.beta_t*kl(*mus_logvars_t)
        reco = mse(image_hat_logits, images)
        #cov = decorrelate_params(*mus_logvars_s, *mus_logvars_t)

        if log_components :
            self.log("loss/kl_s", weighted_kl_s)
            self.log("loss/kl_t", weighted_kl_t)
            self.log("loss/reco", reco)

        return weighted_kl_t+weighted_kl_s+reco
    
    def training_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.forward(images)
        loss = self.loss(mus_logvars_s, mus_logvars_t, image_hat_logits, images, log_components=True)

        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        self.log("loss/train", loss)

        if self.images_train_buff is None : 
            self.images_train_buff = images
            self.image_ref = self.images_train_buff[0]
            self.z_ref = {"s":z_s[0], "t": z_t[0]}

        return loss

    def validation_step(self, batch: tuple[torch.Tensor]):
        images, labels = batch
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.forward(images)
        loss = self.loss(mus_logvars_s, mus_logvars_t, image_hat_logits, images)

        self.log("loss/val", loss)
        print(f"Val loss: {loss}")
    
    def test_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.forward(images, test=True)

        weighted_kl_s = self.hparams.beta_s*kl(*mus_logvars_s)
        weighted_kl_t = self.hparams.beta_t*kl(*mus_logvars_t)
        reco = mse(image_hat_logits, images)

        self.log("loss/reco_test", reco)
        self.log("loss/kl_s_test", weighted_kl_s)
        self.log("loss/kl_t_test", weighted_kl_t)
        self.log("loss/test", reco+weighted_kl_t+weighted_kl_s)

        if self.images_test_buff is None : self.images_test_buff = images

    def on_train_epoch_end(self):
        epoch = self.current_epoch

        if epoch % 10 == 0:
            try : os.mkdir(os.path.join(self.logger.log_dir, f"epoch_{epoch}"))
            except FileExistsError as e : pass

            self.show_reconstruct(self.images_train_buff) #display images and reconstruction in interactive mode; save the plot in plt.gcf() if non interactive
            #save the recontruction plot saved in plt.gcf()
            save_reco_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"reco_{epoch}.png")
            plt.gcf().savefig(save_reco_path)

            #save the generate images
            images_gen = self.generate()
            save_gen_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_{epoch}.png")
            vutils.save_image(images_gen.detach().cpu(), save_gen_path)
             
            #save the generate images
            images_gen = self.generate()
            save_gen_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_{epoch}.png")
            vutils.save_image(images_gen.detach().cpu(), save_gen_path)
              
            #save the cond generate image s
            images_cond_s_gen = self.generate_cond(cond="s")
            images_cond_s_gen_ref = torch.cat([images_cond_s_gen.detach().cpu(), self.image_ref.detach().cpu().unsqueeze(0)])
            save_gen_s_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_s_{epoch}.png")
            vutils.save_image(images_cond_s_gen_ref, save_gen_s_path)

            #save the cond generate image t
            images_cond_t_gen = self.generate_cond(cond="t")
            images_cond_t_gen_ref = torch.cat([images_cond_t_gen.detach().cpu(), self.image_ref.detach().cpu().unsqueeze(0)])
            save_gen_t_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_t_{epoch}.png")
            vutils.save_image(images_cond_t_gen_ref.detach().cpu(), save_gen_t_path)

        self.images_train_buff = None
        self.z_ref = None

    def on_test_end(self):
        images_gen = self.generate()
        labels_gen = torch.zeros([images_gen.shape[0],1])

        display((images_gen.detach().cpu(), labels_gen.detach().to("cpu")))

        self.logger.experiment.add_figure("img/gen", plt.gcf())

        self.show_reconstruct(self.images_test_buff)
        self.logger.experiment.add_figure("img/reco", plt.gcf())

        vutils.save_image(images_gen.detach().cpu(), os.path.join(self.logger.log_dir, "gen.png"))
