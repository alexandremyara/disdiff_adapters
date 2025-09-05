import torch
import torch.nn as nn
from lightning import LightningModule
import matplotlib.pyplot as plt
import os
import math
from os.path import join
import shutil
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

    def forward(self, images: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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
                 select_factor: int=0,
                 res_block: nn.Module=ResidualBlock,
                 beta_s: float=1.0,
                 beta_t: float=1.0,
                 warm_up: bool=False,
                 kl_weight: float= 1e-6,
                 type: str="all",
                 l_cov: float=0.0,
                 l_nce: float=1e-3,
                 l_anti_nce: float=0.0,
                 temp: float=0.07,
                 factor_value: int = -1,
                 map_idx_labels: list|None= None) :
        
        super().__init__()
        self.save_hyperparameters()
        self.model = _MultiDistillMe(in_channels=self.hparams.in_channels,
                                     img_size=self.hparams.img_size,
                                     latent_dim_s=self.hparams.latent_dim_s,
                                     latent_dim_t=self.hparams.latent_dim_t,
                                     res_block=self.hparams.res_block)
            

        self.images_test_buff = None
        self.images_train_buff = []
        self.labels_train_buff = []
        self.latent_train_buff:dict[str, torch.Tensor] = {"s" : [], "t": []}

        self.constrastive = InfoNCESupervised(temperature=self.hparams.temp)
        self.current_batch = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())
    
    def generate(self, nb_samples: int=16) :
        eps_s = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_s])).to(self.device, torch.float32)
        eps_t = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_t])).to(self.device, torch.float32)


        z = self.model.merge_operation(eps_s, eps_t)

        x_hat_logits = self.model.decoder(z)
        return x_hat_logits

    def generate_cond(self, nb_samples: int=16, cond: str="t", pos: int=0, 
                      z_t=None, z_s=None, img_ref=None, factor_value=-1) :
        #Test cond validity and if z_t/z_s are specified properly
        if not isinstance(cond, str) or cond.lower() not in {"s", "t"}:
            raise ValueError(f"cond must be 's' or 't', got {cond!r}")
        assert (z_t is None) == (z_s is None), "You must specified z_s and z_t, or none of them"

        #When z_t/z_s are not specified, we use the buffer
        if z_t is None : 
            z_t= self.latent_train_buff["t"]
            z_s = self.latent_train_buff["s"]
        else : print("interactive mode : on") #If z_t/z_s are specified, mode is not auto

        #If a factor_value should be represented, we mask the latent space to select the correct value
        if factor_value != -1 :
            mask = (self.labels_train_buff[:, self.hparams.select_factor] == factor_value)
            if cond == "s" : z_t = z_t[mask]
            else : z_s = z_s[mask]
        else : mask = torch.ones(z_t.size(0), dtype=bool)

        if cond == "t" :
            nb_sample_latent = z_t.shape[0]
            assert nb_samples <= nb_sample_latent, "Too much points"

            idx_t = torch.randint_like(torch.zeros([nb_samples]), high=nb_sample_latent, dtype=torch.int32)
            eps_t = z_t[idx_t].to(self.device)

            if img_ref is None :
                eps_s = torch.stack(nb_samples*[z_s[pos]]).to(self.device)
            else : 
                with torch.no_grad() : _, _, _, eps_s, _, _ = self(img_ref.to(self.device), test=True)
                eps_s = torch.cat(nb_samples*[eps_s]).to(self.device)

            assert eps_s.device == eps_t.device, "eps_s, eps_t have to be on the same device"
            z = self.model.merge_operation(eps_s, eps_t)

        else :
            nb_sample_latent = z_s.shape[0] 
            assert nb_samples <= nb_sample_latent, "Too much points" 

            idx_s = torch.randint_like(torch.zeros([nb_samples]), high=nb_sample_latent, dtype=torch.int32)
            eps_s = z_s[idx_s].to(self.device)

            if img_ref is None :
                eps_t = torch.stack(nb_samples*[z_t[pos]]).to(self.device)
            else : 
                with torch.no_grad() : _, _, _, _, eps_t, _ = self(img_ref.to(self.device), test=True)
                eps_t = torch.cat(nb_samples*[eps_t]).to(self.device)
            assert eps_t.device == eps_s.device, "eps_s, eps_t have to be on the same device"
            z = self.model.merge_operation(eps_s, eps_t)

        x_hat_logits = self.model.decoder(z)
        ref_img = img_ref if img_ref is not None else self.images_train_buff[mask][pos]
        return x_hat_logits.detach().cpu(), ref_img.unsqueeze(0).detach().cpu()

    def show_reconstruct(self, images: torch.Tensor) :
        images = images.to(self.device)[:8]
        with torch.no_grad() : mus_logvars_s, mus_logvars_t, images_gen, z_s, z_t, z = self(images, test=True)

        fig, axes = plt.subplots(len(images), 2, figsize=(7, 20))

        for i in range(len(images)) :
            img = images[i]
            img_gen = images_gen[i]

            images_proc = (255*((img - img.min()) / (img.max() - img.min() + 1e-8))).to("cpu",torch.uint8).permute(1,2,0).detach().cpu().numpy()
            images_gen_proc = (255*((img_gen - img_gen.min()) / (img_gen.max() - img_gen.min() + 1e-8))).to("cpu",torch.uint8).permute(1,2,0).detach().cpu().numpy()

            axes[i,0].imshow(images_proc)
            axes[i,1].imshow(images_gen_proc)

            axes[i,0].set_title("original")
            axes[i,1].set_title("reco")
        plt.tight_layout()
        plt.show()
        return mus_logvars_s, mus_logvars_t, images_gen, z_s, z_t, z

    def forward(self, images: torch.Tensor, test=False) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        mu_t, logvar_t = mus_logvars_t
        mu_s, logvar_s = mus_logvars_s
        report_nonfinite(mu_t=mu_t, 
                         logvar_t=logvar_t, 
                         mu_s=mu_s,
                         logvar_s=logvar_s,
                         image_hat_logits=image_hat_logits, 
                         images=images, 
                         z_s=z_s,
                         z_t=z_t,
                         labels=labels)
        # assert not torch.isnan(mus_logvars_t[0]).any(), "mu_s is nan"
        # assert not torch.isnan(mus_logvars_t[1]).any(), "sigma_s is nan"
        # assert not torch.isnan(z_t).any(), "z_t is nan"
        # assert not torch.isnan(mus_logvars_t[0]).any(), "mu_s is nan"
        # assert not torch.isnan(mus_logvars_t[1]).any(), "sigma_s is nan"
        # assert not torch.isnan(image_hat_logits).any(), "image_hat_logits is nan"
        


        weighted_kl_s = self.hparams.kl_weight*self.hparams.beta_s*kl(*mus_logvars_s)
        assert not torch.isnan(weighted_kl_s).any(), "kl_s is Nan"
        weighted_kl_t = self.hparams.kl_weight*self.hparams.beta_t*kl(*mus_logvars_t)
        assert not torch.isnan(weighted_kl_t).any(), "kl_t is Nan"
        reco = mse(image_hat_logits, images)
        assert not torch.isnan(reco).any(), "reco is Nan"
        cov = self.hparams.l_cov*decorrelate_params(*mus_logvars_s, *mus_logvars_t)
        assert not torch.isnan(cov).any(), "cov is Nan"
        nce = self.hparams.l_nce*self.constrastive(z_t, labels)
        assert not torch.isnan(nce).any(), "nce is Nan"
        
        if log_components :
            self.log("loss/kl_s", weighted_kl_s.detach())
            self.log("loss/kl_t", weighted_kl_t.detach())
            self.log("loss/reco", reco.detach())
            self.log("loss/cov", cov.detach())
            self.log("loss/nce", nce.detach())

        if self.hparams.type == "all" : 
            if self.hparams.warm_up and self.current_epoch <= int(0.1*self.trainer.max_epochs) :
                loss_value = weighted_kl_t+weighted_kl_s+reco+nce
            else : loss_value = weighted_kl_t+weighted_kl_s+reco+cov+nce
        elif self.hparams.type == "vae" : loss_value = weighted_kl_t+weighted_kl_s+reco
        elif self.hparams.type == "vae_nce" : loss_value = weighted_kl_t+weighted_kl_s+reco+nce
        elif self.hparams.type == "vae_cov" : loss_value = weighted_kl_t+weighted_kl_s+reco+cov
        elif self.hparams.type == "reco" : loss_value = reco
        elif self.hparams.type == "kl" : loss_value =  weighted_kl_t+weighted_kl_s
        elif self.hparams.type == "cov" : loss_value = cov
        elif self.hparams.type == "nce" : loss_value = nce
        else : raise ValueError("Loss type error")

        return loss_value
    
    def training_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.forward(images)
        loss = self.loss(mus_logvars_s, mus_logvars_t, image_hat_logits, images, z_s, z_t, 
                         labels=labels[:, self.hparams.select_factor], log_components=True)
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        self.log("loss/train", loss)

        if self.current_batch <= 700 :
            self.images_train_buff.append(images.detach().cpu())
            self.labels_train_buff.append(labels.detach().cpu())

        self.current_batch += 1

        return loss

    def validation_step(self, batch: tuple[torch.Tensor]):
        images, labels = batch
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.forward(images)
        loss = self.loss(mus_logvars_s, mus_logvars_t, image_hat_logits, images, z_s, z_t, labels=labels[:, 0])
        if torch.isnan(loss):
            raise ValueError("NaN loss")
        self.log("loss/val", loss)
        print(f"Val loss: {loss}")
    
    def test_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.forward(images, test=True)

        weighted_kl_s = self.hparams.beta_s*kl(*mus_logvars_s)
        weighted_kl_t = self.hparams.beta_t*kl(*mus_logvars_t)
        reco = mse(image_hat_logits, images)

        self.log("loss/reco_test", reco, sync_dist=True)
        self.log("loss/kl_s_test", weighted_kl_s, sync_dist=True)
        self.log("loss/kl_t_test", weighted_kl_t, sync_dist=True)
        self.log("loss/test", reco+weighted_kl_t+weighted_kl_s, sync_dist=True)

        if self.images_test_buff is None : self.images_test_buff = images

    def on_train_epoch_start(self):
        self.images_train_buff = []
        self.labels_train_buff = []
        self.latent_train_buff = {"s" : [], "t": []}
        self.current_batch = 0

    def on_train_epoch_end(self):
        epoch = self.current_epoch

        if epoch % 2 == 0:
            try : os.mkdir(os.path.join(self.logger.log_dir, f"epoch_{epoch}"))
            except FileExistsError as e : pass

            #compute a sample of the latent space
            for images in self.images_train_buff :
                with torch.no_grad() : _, _, _, z_s, z_t, z = self.forward(images.to(self.device), test=True) #images shape : [32, 3, 64, 64]

                self.latent_train_buff["s"].append(z_s.detach().cpu())
                self.latent_train_buff["t"].append(z_t.detach().cpu())
            self.latent_train_buff["s"] = torch.cat(self.latent_train_buff["s"])
            self.latent_train_buff["t"] = torch.cat(self.latent_train_buff["t"])

            self.images_train_buff = torch.cat(self.images_train_buff)
            self.labels_train_buff = torch.cat(self.labels_train_buff)
            mus_logvars_s, mus_logvars_t = self.log_reco()

            self.log_gen_images()

            path_heatmap = join(self.logger.log_dir, f"epoch_{epoch}", f"cov_{epoch}.png")
            log_cross_cov_heatmap(*mus_logvars_s, *mus_logvars_t, save_path=path_heatmap)

            ### latent space
            self.log_latent()

    def reload_latent(self):
        self.on_train_epoch_start()
        self.images_train_buff = torch.load(Shapes3D.Path.BUFF_IMG)
        self.labels_train_buff = torch.load(Shapes3D.Path.BUFF_LABELS)

        if type(self.images_train_buff) == list : self.images_train_buff = torch.cat(self.images_train_buff)
        if type(self.labels_train_buff) == list : self.labels_train_buff = torch.cat(self.labels_train_buff)
        
        pill = []
        for images in self.images_train_buff :
            if len(pill) < 4096 : pill.append(images)
            else : 
                pill.append(images)
                images_batched = torch.stack(pill)
                print(images_batched.shape)
                with torch.no_grad() : _, _, _, z_s, z_t, z = self.forward(images_batched.to(self.device), test=True) #images shape : [32, 3, 64, 64]
                pill = []
                self.latent_train_buff["s"].append(z_s.detach().cpu())
                self.latent_train_buff["t"].append(z_t.detach().cpu())

        images_batched = torch.stack(pill)
        print(images_batched.shape)
        with torch.no_grad() : _, _, _, z_s, z_t, z = self.forward(images_batched.to(self.device), test=True) #images shape : [32, 3, 64, 64]
        pill = []
        self.latent_train_buff["s"].append(z_s.detach().cpu())
        self.latent_train_buff["t"].append(z_t.detach().cpu())

        self.latent_train_buff["s"] = torch.cat(self.latent_train_buff["s"])
        self.latent_train_buff["t"] = torch.cat(self.latent_train_buff["t"])

    def on_test_end(self):
        images_gen = self.generate()
        labels_gen = torch.zeros([images_gen.shape[0],1])

        display((images_gen.detach().cpu(), labels_gen.detach().to("cpu")))

        self.logger.experiment.add_figure("img/gen", plt.gcf())

        self.show_reconstruct(self.images_test_buff)
        self.logger.experiment.add_figure("img/reco", plt.gcf())

        vutils.save_image(images_gen.detach().cpu(), os.path.join(self.logger.log_dir, "gen.png"))

    def log_reco(self) :
        mus_logvars_s, mus_logvars_t, images_gen, z_s, z_t, z = self.show_reconstruct(self.images_train_buff[:8]) #display images and reconstruction in interactive mode; save the plot in plt.gcf() if non interactive
        #save the recontruction plot saved in plt.gcf()
        save_reco_path = os.path.join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"reco_{self.current_epoch}.png")
        fig = plt.gcf()
        fig.savefig(save_reco_path)
        plt.close(fig)
        return mus_logvars_s, mus_logvars_t

    def log_gen_images(self) :
        epoch = self.current_epoch

        #save the generate images
        images_gen = self.generate()
        save_gen_path = join(self.logger.log_dir, f"epoch_{epoch}", f"gen_{epoch}.png")
        vutils.save_image(images_gen.detach().cpu(), save_gen_path)
            
        #save the cond generate image s
        for i in range(4) :
            factor_value = self.hparams.factor_value if i%2 else -1
            images_cond_s_gen, input_t = self.generate_cond(cond="s", pos=i, factor_value=factor_value)
            images_cond_s_gen_ref = torch.cat([images_cond_s_gen.detach().cpu(), input_t])
            save_gen_s_path = join(self.logger.log_dir, f"epoch_{epoch}", f"gen_s_{epoch}_{i}.png")
            vutils.save_image(images_cond_s_gen_ref, save_gen_s_path)

        #save the cond generate image t
        for i in range(4) :
            factor_value = self.hparams.factor_value if i%2 else -1
            images_cond_t_gen, input_s = self.generate_cond(cond="t", pos=i, factor_value=factor_value)
            images_cond_t_gen_ref = torch.cat([images_cond_t_gen.detach().cpu(), input_s])
            save_gen_t_path = join(self.logger.log_dir, f"epoch_{epoch}", f"gen_t_{epoch}_{i}.png")
            vutils.save_image(images_cond_t_gen_ref.detach().cpu(), save_gen_t_path)

        ### Merge in one image
        final_gen_s = merge_images_with_black_gap(
                                    [ join(self.logger.log_dir, f"epoch_{epoch}", f"gen_s_{epoch}_{i}.png") for i in range(4)]
                                    )
        final_gen_t = merge_images_with_black_gap(
                        [ join(self.logger.log_dir, f"epoch_{epoch}", f"gen_t_{epoch}_{i}.png") for i in range(4)]
                        )
        final_gen_s.save(join(self.logger.log_dir, "final_gen_s.png"))
        final_gen_t.save(join(self.logger.log_dir, "final_gen_t.png"))
        final_image = merge_images(save_gen_path, join(self.logger.log_dir, "final_gen_s.png"), join(self.logger.log_dir, "final_gen_t.png"))
        save_gen_all_path = join(self.logger.log_dir, f"epoch_{epoch}", f"gen_all_{epoch}.png")
        final_image.save(save_gen_all_path)

        os.remove(join(self.logger.log_dir, "final_gen_s.png"))
        os.remove(join(self.logger.log_dir, "final_gen_t.png"))
        for i in range(4) : 
            os.remove(join(self.logger.log_dir, f"epoch_{epoch}", f"gen_s_{epoch}_{i}.png"))
            os.remove(join(self.logger.log_dir, f"epoch_{epoch}", f"gen_t_{epoch}_{i}.png"))

    def log_latent(self) :
        number_labels = self.labels_train_buff.shape[1]
        
        for i in range(number_labels) :
            labels = self.labels_train_buff[:, i].unsqueeze(1)
            z_s_path = join(self.logger.log_dir, "z_s.png")
            z_t_path = join(self.logger.log_dir, "z_t.png")
            title = f"latent space {i}" if self.hparams.map_idx_labels is None else self.hparams.map_idx_labels[i]

            display_latent(labels=labels, z=self.latent_train_buff["s"], title=title)
            fig = plt.gcf()
            fig.savefig(z_s_path)
            plt.close(fig)

            display_latent(labels=labels, z=self.latent_train_buff["t"], title=title)
            fig = plt.gcf()
            fig.savefig(z_t_path)
            plt.close(fig)
            latent_img = merge_images_with_black_gap([z_s_path, z_t_path])
            latent_img.save(join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"latent_space_{i}_{self.current_epoch}.png"))
            os.remove(z_s_path)
            os.remove(z_t_path)

        grid_merge([join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"latent_space_{i}_{self.current_epoch}.png") for i in range(number_labels)], join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"latent_space_{self.current_epoch}.png"))
        for i in range(number_labels) : os.remove(join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"latent_space_{i}_{self.current_epoch}.png"))
