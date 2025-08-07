import torch
from torch.utils.data import Dataset
from lightning import LightningModule
from disdiff_adapters.arch.multi_distillme import *

class LatentDataset(Dataset) :

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, Model_class: LightningModule=MultiDistillMeModule, path: str="/projects/compures/alexandre/disdiff_adapters/disdiff_adapters/logs/md/shapes/test_dim_s2(1)/md_epoch=30_beta=(1.0, 1.0)_latent=(2, 2)_batch=32_warm_up=True_lr=1e-05_arch=res+l_cov=0.0+l_nce=0.1+l_anti_nce=0.0/checkpoints/epoch=18-step=182400.ckpt") :
        super().__init__()
        self.Model_class = Model_class
        self.path = path
        self.model = self.load_ckpt()
        self.images = images
        self.labels = labels

        assert images.shape[0] == labels.shape[0], "images and labels should have the same shape[0]"

    def load_ckpt(self) -> LightningModule :
        return self.Model_class.load_from_checkpoint(self.path)
    
    def __len__(self) -> int :
        return self.images.shape[0]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        images = self.images[index]
        if type(images) != torch.Tensor : images = torch.tensor(images)

        processed_img = (images.permute(2,0,1)/255).to(torch.float32)
        with torch.no_grad() :
            processed_img = processed_img.to(self.model.device) 
            mus_logvars_s, mus_logvars_t, image_hat_logits, z_s, z_t, z = self.model(processed_img.unsqueeze(0))
        return z_s.squeeze(0), z_t.squeeze(0), z.squeeze(0), self.labels[index]