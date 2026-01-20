import argparse
import logging
import os
from dataclasses import fields

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn

from disdiff_adapters.arch.multi_distillme import Xfactors
from disdiff_adapters.arch.vae.block import ResidualBlock, SimpleConv
from disdiff_adapters.data_module import (
    Cars3DDataModule,
    CelebADataModule,
    DSpritesDataModule,
    MPI3DDataModule,
    Shapes3DDataModule,
)
from disdiff_adapters.metric.callbacks import DisentanglementMetricsCallback
from disdiff_adapters.utils.const import (
    LOG_DIR,
    LOG_DIR_SHELL,
    MPI3D,
    Cars3D,
    CelebA,
    DSprites,
    Shapes3D,
)

SEED = 2025

torch.set_float32_matmul_precision("high")


def setup_logging(log_dir: str, log_filename: str = "train.log") -> None:
    """Configure logging to both console and file.

    Sets up the root logger and Lightning's internal logger to write
    to both stderr and a file in the experiment directory.
    """
    log_path = os.path.join(log_dir, log_filename)
    os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    # Also configure Lightning's logger to use our handlers
    lightning_logger = logging.getLogger("lightning.pytorch")
    lightning_logger.setLevel(logging.INFO)
    lightning_logger.handlers.clear()
    lightning_logger.addHandler(console_handler)
    lightning_logger.addHandler(file_handler)

    rank_zero_info(f"Logging to file: {log_path}")


def to_list(x: str) -> list[int]:
    return [int(elt) for elt in x.split(",")]


def to_list_float(x: str) -> list[float]:
    return [float(elt) for elt in x.split(",")]


def str_to_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "t"}


def list_to_str(l: list) -> str:
    s = ""
    for elt in l:
        s += str(elt)
        s += ","
    return s[:-1]


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
        "--dataset", type=str, help="dataset name used.", default="bloodmnist"
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Max number of epochs.",
        default=50,
    )

    parser.add_argument("--batch_size", type=int, help="batch size", default=32)

    parser.add_argument("--warm_up", type=str, default="False")

    parser.add_argument("--lr", type=float, default=10e-5, help="learning rate.")

    parser.add_argument(
        "--beta_s",
        type=float,
        help="beta used",
        default=1.0,
    )

    parser.add_argument(
        "--beta_t",
        type=float,
        help="beta used",
        default=1.0,
    )

    parser.add_argument(
        "--latent_dim_s", type=int, help="dimension of the latent space", default=4
    )

    parser.add_argument(
        "--dims_by_factors",
        type=to_list,
        help="dimension of the latent space t",
        default="2",
    )

    parser.add_argument(
        "--factor_value", type=int, default=1, help="Choose a factor to encode"
    )

    parser.add_argument(
        "--arch", type=str, default="def", help="main value of the interest factor"
    )

    parser.add_argument("--loss_type", type=str, default="all", help="select loss type")

    parser.add_argument("--l_cov", type=float, default=0, help="use cross cov loss")

    parser.add_argument(
        "--l_nce_by_factors", type=to_list_float, default="0.1", help="use nce loss"
    )

    parser.add_argument("--l_anti_nce", type=float, default=0, help="use anti nce loss")

    parser.add_argument(
        "--experience", type=str, default="", help="Name of the experience"
    )

    parser.add_argument("--key", type=str, default="", help="key to add for the file")

    parser.add_argument(
        "--gpus", type=to_list, default="0", help="comma seperated list of gpus"
    )

    parser.add_argument("--version_model", type=str, default="debug")

    parser.add_argument(
        "--kl_weight_scale",
        type=float,
        default=1.0,
        help="Multiplicative scale on KL weight (keeps dataset's default when == 1.0)",
    )

    parser.add_argument(
        "--wandb", type=str, default="False", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases entity/team"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name (defaults to version string)",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated list of Weights & Biases tags",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        help="W&B mode override (online/offline/disabled)",
    )
    parser.add_argument(
        "--compute_metrics",
        type=str,
        default="True",
        help="Whether to compute disentanglement metrics (FactorVAE, DCI) during training",
    )
    parser.add_argument(
        "--metric_interval",
        type=int,
        default=5,
        help="Compute metrics every N epochs (default: 5)",
    )
    parser.add_argument(
        "--metric_n_iter",
        type=int,
        default=153600,
        help="Number of iterations for FactorVAE metric (default: 153600)",
    )
    return parser.parse_args()


def main(flags: argparse.Namespace):
    warm_up = str_to_bool(flags.warm_up)
    use_wandb = str_to_bool(flags.wandb)
    compute_metrics = str_to_bool(flags.compute_metrics)
    kl_weight_scale = flags.kl_weight_scale

    res_block = ResidualBlock if flags.arch == "res" else SimpleConv

    # Load data_module
    match flags.dataset:
        case "dsprites":
            data_module = DSpritesDataModule(batch_size=flags.batch_size)
            param_class = DSprites
            in_channels = 1
            img_size = 64
            klw = 0.000001
            factor_value = -1
            factor_value_1 = -1
            select_factors = [k for k in range(5 - 1)]
            binary_factor = False

        case "mpi3d":
            data_module = MPI3DDataModule(batch_size=flags.batch_size)
            param_class = MPI3D
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = -1
            factor_value_1 = -1
            select_factors = [k for k in range(7 - 1)]
            binary_factor = False

        case "shapes":
            data_module = Shapes3DDataModule(batch_size=flags.batch_size)
            param_class = Shapes3D
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = -1
            factor_value_1 = -1
            select_factors = [k for k in range(6 - 1)]
            binary_factor = False

        case "celeba":
            data_module = CelebADataModule(batch_size=flags.batch_size)
            param_class = CelebA
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = 1
            factor_value_1 = 0
            select_factors = CelebA.Params.REPRESENTANT_IDX
            binary_factor = True

        case "cars3d":
            data_module = Cars3DDataModule(batch_size=flags.batch_size)
            param_class = Cars3D
            in_channels = 3
            img_size = 128
            klw = 0.000001
            factor_value = -1
            factor_value_1 = -1
            select_factors = [k for k in range(3 - 1)]
            binary_factor = False

        case _:
            raise ValueError("Error flags.dataset")
    rank_zero_info(f"Using dataset: {param_class.__name__}")
    rank_zero_info("Params:")
    for f in fields(param_class.Params):
        rank_zero_info(f"  {f.name}: {getattr(param_class.Params, f.name)}")

    num_factors = len(select_factors)

    # dim_S and dim_T set here via CLI args
    user_dims = flags.dims_by_factors
    if len(user_dims) == 1:
        dims_by_factors = [user_dims[0]] * num_factors
    elif len(user_dims) == num_factors:
        dims_by_factors = user_dims
    else:
        raise ValueError(
            f"dims_by_factors must be len 1 or {num_factors}, got {len(user_dims)}"
        )

    user_l_nce = flags.l_nce_by_factors
    if len(user_l_nce) == 1:
        l_nce_by_factors = [user_l_nce[0]] * num_factors
    elif len(user_l_nce) == num_factors:
        l_nce_by_factors = user_l_nce
    else:
        raise ValueError(
            f"l_nce_by_factors must be len 1 or {num_factors}, got {len(user_l_nce)}"
        )

    klw = klw * kl_weight_scale

    map_idx_labels = param_class.Params.FACTORS_IN_ORDER
    rank_zero_info(f"dims_by_factors: {dims_by_factors}")
    model = Xfactors(
        in_channels=in_channels,
        img_size=img_size,
        latent_dim_s=flags.latent_dim_s,
        dims_by_factors=dims_by_factors,
        select_factors=select_factors,
        factor_value=factor_value,
        factor_value_1=factor_value_1,
        res_block=res_block,
        beta_s=flags.beta_s,
        beta_t=flags.beta_t,
        warm_up=warm_up,
        kl_weight=klw,
        type=flags.loss_type,
        l_cov=flags.l_cov,
        l_nce_by_factors=l_nce_by_factors,
        l_anti_nce=flags.l_anti_nce,
        map_idx_labels=map_idx_labels,
        temp=0.03,
        binary_factor=binary_factor,
    )

    # Use experience from CLI (set by train_x.sh) as the canonical version string
    assert flags.experience is not None and flags.experience != "", (
        f"experience must be set, got '{flags.experience}'"
    )
    version = flags.experience
    rank_zero_info(f"VERSION : {version}")
    rank_zero_info(f"Select factor : {select_factors}, factor value : {factor_value}")

    # LOG_DIR is already the full target directory (set by train_x.sh)
    tb_logger = TensorBoardLogger(
        save_dir=LOG_DIR,
        name="",
        version="",
        default_hp_metric=False,
    )

    # Set up file logging now that we know the log directory
    setup_logging(tb_logger.log_dir)

    ckpt_dir = os.path.join(tb_logger.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="loss/val",
        mode="min",
        save_top_k=1,  # garde uniquement le meilleur
        save_last=True,  # en plus, maintient checkpoints/last.ckpt (dernier)
        filename="best-{epoch:03d}",  # le nom du "best" (Lightning ajoutera la métrique)
    )

    loggers = [tb_logger]

    if use_wandb:
        wandb_tags = [tag for tag in flags.wandb_tags.split(",") if tag]
        wandb_kwargs = {
            "project": flags.wandb_project,
            "entity": flags.wandb_entity,
            "name": flags.wandb_run_name or version,
            "dir": LOG_DIR_SHELL,
            "config": vars(flags),
            "tags": wandb_tags or None,
            "log_model": False,
        }
        if flags.wandb_mode:
            wandb_kwargs["mode"] = flags.wandb_mode

        # drop keys with None to keep wandb.init clean
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}

        wandb_logger = WandbLogger(**wandb_kwargs)
        loggers.append(wandb_logger)

    callbacks = [
        ckpt_cb,
        LearningRateMonitor("epoch"),
    ]

    # Add disentanglement metrics callback (only for supported datasets)
    if compute_metrics:
        match flags.dataset:
            case "shapes":
                callbacks.append(
                    DisentanglementMetricsCallback(
                        compute_every_n_epochs=flags.metric_interval,
                        n_iter=flags.metric_n_iter,
                        batch_size=64,
                        verbose=True,
                    )
                )
            case _:
                rank_zero_warn(
                    f"Disentanglement metrics not implemented for dataset '{flags.dataset}', skipping."
                )

    trainer = Trainer(
        accelerator="auto",
        devices=flags.gpus,
        gradient_clip_val=3.0,
        max_epochs=flags.max_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,  # val à chaque époque => last.ckpt se met à jour à chaque epoch
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, data_module)

    # Run test phase after training
    trainer.test(model, data_module)


if __name__ == "__main__":
    flags = parse_args()

    rank_zero_info("Parameters:")
    for k, v in sorted(vars(flags).items()):
        rank_zero_info(f"{k:20s}: {v}")

    main(flags)
