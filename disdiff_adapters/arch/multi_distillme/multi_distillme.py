import torch
import torch.nn as nn
from lightning import LightningModule
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils
from PIL import Image
import numpy as np

from disdiff_adapters.arch.vae import *
from disdiff_adapters.utils import *
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
        self.latent_buff = {"s":[], "t":[]}

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
                 l_cov: float=1,
                 l_nce: float=1,
                 l_anti_nce: float=1,) :
        
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
        self.constrastive = InfoNCESupervised()
        self.latent_buff = self.model.latent_buff
        self.labels_buff = self.model.labels_buff
        self.current_batch = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())
    
    def generate(self, nb_samples: int=16) :
        eps_s = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_s])).to(self.device, torch.float32)
        eps_t = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_t])).to(self.device, torch.float32)


        z = self.model.merge_operation(eps_s, eps_t)

        x_hat_logits = self.model.decoder(z)
        return x_hat_logits

    def generate_cond(self, nb_samples: int=16, cond: str="t", pos: int=0) :
        if cond == "t" :
            eps_t = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_t])).to(self.device, torch.float32)
            z_s = torch.stack(nb_samples*[self.z_ref["s"][pos]])
            z = self.model.merge_operation(z_s, eps_t)
        elif cond == "s" :
            eps_s = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_s])).to(self.device, torch.float32)
            z_t = torch.stack(nb_samples*[self.z_ref["t"][pos]])
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
        return mus_logvars_s, mus_logvars_t, images_gen, z_s, z_t, z

    def forward(self, images: torch.Tensor, test=False) :
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.model(images)

        return mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z
    
    def loss(self, mus_logvars_s: torch.Tensor, 
             mus_logvars_t: torch.Tensor, 
             image_hat_logits: torch.Tensor, 
             images: torch.Tensor, 
             z_s: torch.Tensor,
             z_t: torch.Tensor,
             labels=None, 
             log_components: bool=False) :

        weighted_kl_s = self.hparams.kl_weight*self.hparams.beta_s*kl(*mus_logvars_s)
        weighted_kl_t = self.hparams.kl_weight*self.hparams.beta_t*kl(*mus_logvars_t)
        reco = mse(image_hat_logits, images)
        #cov = decorrelate_params(*mus_logvars_s, *mus_logvars_t, l_var=0)/(self.hparams.latent_dim_s*self.hparams.latent_dim_t)
        nce = self.constrastive(z_t, labels)

        
        if log_components :
            self.log("loss/kl_s", weighted_kl_s)
            self.log("loss/kl_t", weighted_kl_t)
            self.log("loss/reco", reco)
            #self.log("loss/cov", cov)
            self.log("loss/nce", nce)

        #loss_value = weighted_kl_t+weighted_kl_s+reco+self.hparams.l_cov*cov+self.hparams.l_nce*nce
        #loss_value = weighted_kl_t+weighted_kl_s+reco
        loss_value = weighted_kl_t+weighted_kl_s+reco+self.hparams.l_nce*nce
        return loss_value
    
    def training_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch

        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.forward(images)
        loss = self.loss(mus_logvars_s, mus_logvars_t, image_hat_logits, images, z_s, z_t, 
                         labels=labels[:, 0], log_components=True)

        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        self.log("loss/train", loss)

        if self.images_train_buff is None : 
            self.images_train_buff = images
            self.labels_train_buff = labels
        if self.current_batch <= 700 :   
            self.labels_buff.append(labels[:,0].unsqueeze(1))
            self.latent_buff["s"].append(z_s)
            self.latent_buff["t"].append(z_t)
        self.current_batch += 1
        return loss

    def validation_step(self, batch: tuple[torch.Tensor]):
        images, labels = batch
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.forward(images)
        loss = self.loss(mus_logvars_s, mus_logvars_t, image_hat_logits, images, z_s, z_t, labels=labels[:, 0])

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
        self.current_batch = 0

        self.labels_buff = torch.cat(self.labels_buff)
        self.latent_buff["s"] = torch.cat(self.latent_buff["s"])
        self.latent_buff["t"] = torch.cat(self.latent_buff["t"])

        if epoch % 1 == 0:
            try : os.mkdir(os.path.join(self.logger.log_dir, f"epoch_{epoch}"))
            except FileExistsError as e : pass

            #set z_ref : use 32 first images given by the loader - used for evaluate reco and gen
            _, _, _, z_s, z_t, z = self.forward(self.images_train_buff, test=True)
            self.z_ref = {"s":z_s, "t": z_t}


            mus_logvars_s, mus_logvars_t, images_gen, z_s, z_t, z = self.show_reconstruct(self.images_train_buff) #display images and reconstruction in interactive mode; save the plot in plt.gcf() if non interactive
            #save the recontruction plot saved in plt.gcf()
            save_reco_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"reco_{epoch}.png")
            fig = plt.gcf()
            fig.savefig(save_reco_path)
            plt.close(fig)

            self.log_gen_images()

            ### heatmap
            path_heatmap = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"cov_{epoch}.png")
            log_cross_cov_heatmap(*mus_logvars_s, *mus_logvars_t, save_path=path_heatmap)

            ### latent space
            labels = self.labels_train_buff

            display_latent(labels=self.labels_buff, z=self.latent_buff["s"])
            fig = plt.gcf()
            fig.savefig("z_s.png")
            plt.close(fig)

            display_latent(labels=self.labels_buff, z=self.latent_buff["t"])
            fig = plt.gcf()
            fig.savefig("z_t.png")
            plt.close(fig)
            latent_img = merge_images_with_black_gap(["z_s.png", "z_t.png"])
            latent_img.save(os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"latent_space_{epoch}.png"))

        self.images_train_buff = None
        self.labels_train_buff = None
        self.z_ref = None

        self.latent_buff = {"s":[], "t":[]}
        self.labels_buff = []

    def on_test_end(self):
        images_gen = self.generate()
        labels_gen = torch.zeros([images_gen.shape[0],1])

        display((images_gen.detach().cpu(), labels_gen.detach().to("cpu")))

        self.logger.experiment.add_figure("img/gen", plt.gcf())

        self.show_reconstruct(self.images_test_buff)
        self.logger.experiment.add_figure("img/reco", plt.gcf())

        vutils.save_image(images_gen.detach().cpu(), os.path.join(self.logger.log_dir, "gen.png"))

    def log_gen_images(self) :
        epoch = self.current_epoch

        #save the generate images
        images_gen = self.generate()
        save_gen_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_{epoch}.png")
        vutils.save_image(images_gen.detach().cpu(), save_gen_path)
            
        #save the cond generate image s
        for i in range(4) :
            images_cond_s_gen = self.generate_cond(cond="s", pos=i)
            images_cond_s_gen_ref = torch.cat([images_cond_s_gen.detach().cpu(), self.images_train_buff[i].detach().cpu().unsqueeze(0)])
            save_gen_s_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_s_{epoch}_{i}.png")
            vutils.save_image(images_cond_s_gen_ref, save_gen_s_path)

        #save the cond generate image t
        for i in range(4) :
            images_cond_t_gen = self.generate_cond(cond="t", pos=i)
            images_cond_t_gen_ref = torch.cat([images_cond_t_gen.detach().cpu(), self.images_train_buff[i].detach().cpu().unsqueeze(0)])
            save_gen_t_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_t_{epoch}_{i}.png")
            vutils.save_image(images_cond_t_gen_ref.detach().cpu(), save_gen_t_path)

        ### Merge in one image
        final_gen_s = merge_images_with_black_gap(
                                    [ os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_s_{epoch}_{i}.png") for i in range(4)]
                                    )
        final_gen_t = merge_images_with_black_gap(
                        [ os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_t_{epoch}_{i}.png") for i in range(4)]
                        )
        final_gen_s.save("final_gen_s.png")
        final_gen_t.save("final_gen_t.png")
        final_image = merge_images(save_gen_path, "final_gen_s.png", "final_gen_t.png")
        save_gen_all_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_all_{epoch}.png")
        final_image.save(save_gen_all_path)

        os.remove("final_gen_s.png")
        os.remove("final_gen_t.png")
        for i in range(4) : 
            os.remove(os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_s_{epoch}_{i}.png"))
            os.remove(os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_t_{epoch}_{i}.png"))
