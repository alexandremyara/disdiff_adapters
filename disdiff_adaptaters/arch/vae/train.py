import argparse
import torch
from lightning import LightningModule
import matplotlib.pyplot as plt

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer

from disdiff_adaptaters.arch.vae import *
from disdiff_adaptaters.utils import *
from disdiff_adaptaters.loss import *
from disdiff_adaptaters.data_module import *

SEED = 2025


def parse_args() -> argparse.Namespace:
    """
    epochs : max_epoch (int)
    loss : loss name (str)
    optim : optimizer (str)
    arch : model used (str)
    dataset : data module loaded (str)
    pretrained : is already loaded (bool)

    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Max number of epochs.",
        default=50,
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        help="beta used",
        default=1.0,
    )    

    parser.add_argument(
        "--latent_dim",
        type=int,
        help="dimension of the latent space",
        default=4
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name used.",
        default="bloodmnist"
    )

    parser.add_argument(
        "--is_vae",
        type=str,
        help="is a vae",
        default=True
    )

    return parser.parse_args()

def main(flags: argparse.Namespace) :
    device = set_device()
    is_vae = True if flags.is_vae == "True" else False
    # Seed
    L.seed_everything(SEED)
    
    # Load data_module
    match flags.dataset:
        case "bloodmnist":
            data_module = BloodMNISTDataModule(batch_size=16)
            in_channels = 3
            img_size = 28

        case "shapes":
            data_module = Shapes3DDataModule()
            in_channels = 3
            img_size = 64
        case _ :
            raise ValueError("Error flags.dataset")

    L.seed_everything(SEED)

    if is_vae :
        print("\nVAE module\n") 
        model = VAEModule(in_channels = in_channels,
                    img_size=img_size,
                    latent_dim=flags.latent_dim)
        model_name = "vae"
    else :
        print("\nAE module\n") 
        model = AEModule(in_channels = in_channels,
                    img_size=img_size,
                    latent_dim=flags.latent_dim)
        model_name = "ae"
    
    version=f"{model_name}_epoch={flags.max_epochs}_beta={flags.beta}_latent={flags.latent_dim}"
    print(f"\nVERSION : {version}\n")

    trainer = Trainer(
            accelerator="auto",
            devices=[0],

            max_epochs=flags.max_epochs,

            logger=TensorBoardLogger(
                save_dir=LOG_DIR+f"/{model_name}",
                name=flags.dataset,
                version=version,
                default_hp_metric=False,
            ),
            callbacks=[
                ModelCheckpoint(monitor="loss/val", mode="min"),
                LearningRateMonitor("epoch"),
            ]
        )
    
    trainer.fit(model, data_module)


if __name__ == "__main__":
    flags = parse_args()
    main(flags)