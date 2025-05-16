import argparse
import torch
from lightning import LightningModule
import matplotlib.pyplot as plt
import glob

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
import lightning as L

from disdiff_adaptaters.arch.vae import *
from disdiff_adaptaters.utils import *
from disdiff_adaptaters.loss import *
from disdiff_adaptaters.data_module import *
from disdiff_adaptaters.metric import *

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
        default="True")

    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        default=10
    )

    parser.add_argument(
        "--warm_up",
        type=str,
        default="False"
    )

    return parser.parse_args()

def main(flags: argparse.Namespace) :
    device = set_device()
    is_vae = True if flags.is_vae=="True" else False
    warm_up = True if flags.warm_up == "True" else False

    print("\n\nYOU ARE LOADING A VAE\n\n")
    # Seed
    
    # Load data_module
    match flags.dataset:
        case "bloodmnist":
            data_module = BloodMNISTDataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 28

        case "shapes":
            data_module = Shapes3DDataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 64
        case _ :
            raise ValueError("Error flags.dataset")

    callbacks = []

    if is_vae :
        model_class = VAEModule
        model_name = "vae"
        callbacks.append(FIDCallback())
    else :
        model_class = AEModule
        model_name = "ae"

    version=f"vae_epoch={flags.max_epochs}_beta={flags.beta}_latent={flags.latent_dim}_warm_up={warm_up}"
    ckpt_path = glob.glob(f"{LOG_DIR}/{model_name}/{flags.dataset}/{version}/checkpoints/*.ckpt")[0]
    model = model_class.load_from_checkpoint(ckpt_path)
    

    print(f"\nVERSION : {version}\n")

    trainer = Trainer(
            accelerator="auto",
            devices=[1],

            max_epochs=flags.max_epochs,

            logger=TensorBoardLogger(
                save_dir=LOG_DIR+f"/{model_name}",
                name=flags.dataset,
                version=version,
                default_hp_metric=False,
            ),
            callbacks=callbacks
        )
    
    trainer.test(model, data_module)
    

if __name__ == "__main__":
    flags = parse_args()
    main(flags)