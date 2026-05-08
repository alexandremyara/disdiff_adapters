from os.path import join
from os import getenv
from dataclasses import dataclass
from pathlib import Path

# General Const (can be overridden with env vars)
PROJECT_PATH = getenv("PROJECT_PATH", str(Path(__file__).resolve().parents[2]))
DATA_ROOT = getenv("DATA_ROOT", join(PROJECT_PATH, "disdiff_adapters/data"))
LOG_DIR = getenv("LOG_DIR", join(PROJECT_PATH, "logs"))
CELEBA_DATA_DIR = getenv("CELEBA_DATA_DIR", join(DATA_ROOT, "celeba"))
LOG_DIR_SHELL = getenv("LOG_DIR_SHELL")


# Const by Dataset/Model
@dataclass
class Shapes3D:
    @dataclass
    class Path:
        ROOT = join(DATA_ROOT, "3dshapes")
        H5 = join(ROOT, "3dshapes.h5")
        TRAIN = join(ROOT, "shapes3d_train.npz")
        VAL = join(ROOT, "shapes3d_val.npz")
        TEST = join(ROOT, "shapes3d_test.npz")
        NPZ = join(ROOT, "split_3dshapes.npz")
        BUFF_IMG = join(ROOT, "images_train_buff.pt")
        BUFF_LABELS = join(ROOT, "labels_train_buff.pt")

    @dataclass
    class Params:
        FACTORS_IN_ORDER = [
            "floor_hue",
            "wall_hue",
            "object_hue",
            "scale",
            "shape",
            "orientation",
        ]

        NUM_VALUES_PER_FACTOR = {
            "floor_hue": 10,
            "wall_hue": 10,
            "object_hue": 10,
            "scale": 8,
            "shape": 4,
            "orientation": 15,
        }


@dataclass
class MPI3D:
    @dataclass
    class Path:
        ROOT = join(DATA_ROOT, "mpi3d")
        H5 = join(ROOT, "mpi3d.h5")
        NPZ = join(ROOT, "mpi3d_toy.npz")
        TRAIN = join(ROOT, "mpi3d_train.npz")
        VAL = join(ROOT, "mpi3d_val.npz")
        TEST = join(ROOT, "mpi3d_test.npz")

    @dataclass
    class Params:
        FACTORS_IN_ORDER = [
            "object_color",
            "object_shape",
            "object_size",
            "camera_height",
            "background_color",
            "horizontal_axis",
            "vertical_axis",
        ]

        NUM_VALUES_PER_FACTOR = {
            "object_color": 6,
            "object_shape": 6,
            "object_size": 2,
            "camera_height": 3,
            "background_color": 3,
            "horizontal_axis": 40,
            "vertical_axis": 40,
        }


@dataclass
class BloodMNIST:
    @dataclass
    class Path:
        ROOT = join(DATA_ROOT, "bloodmnist")
        TRAIN = join(ROOT, "bloodmnist_train.pt")
        VAL = join(ROOT, "bloodmnist_val.pt")
        TEST = join(ROOT, "bloodmnist_test.pt")
        NPZ = join(ROOT, "bloodmnist.npz")
        H5 = join(ROOT, "bloodmnist.h5")
        VAE = join(LOG_DIR, "vae/bloodmnist")


@dataclass
class CelebA:
    @dataclass
    class Path:
        ROOT = CELEBA_DATA_DIR
        H5 = join(ROOT, "celeba.h5")
        TRAIN = join(ROOT, "celeba_train.npz")
        VAL = join(ROOT, "celeba_val.npz")
        TEST = join(ROOT, "celeba_test.npz")
        NPZ = join(ROOT, "celeba.npz")
        BUFF_IMG = join(ROOT, "images_train_buff.pt")
        BUFF_LABELS = join(ROOT, "labels_train_buff.pt")

    class Params:
        # fmt: off
        FACTORS_IN_ORDER = [
        "5_o_Clock_Shadow",   # 0
        "Arched_Eyebrows",    # 1
        "Attractive",         # 2
        "Bags_Under_Eyes",    # 3
        "Bald",               # 4
        "Bangs",              # 5
        "Big_Lips",           # 6
        "Big_Nose",           # 7
        "Black_Hair",         # 8
        "Blond_Hair",         # 9
        "Blurry",             # 10
        "Brown_Hair",         # 11
        "Bushy_Eyebrows",     # 12
        "Chubby",             # 13
        "Double_Chin",        # 14
        "Eyeglasses",         # 15
        "Goatee",             # 16
        "Gray_Hair",          # 17
        "Heavy_Makeup",       # 18
        "High_Cheekbones",    # 19
        "Male",               # 20
        "Mouth_Slightly_Open",# 21
        "Mustache",           # 22
        "Narrow_Eyes",        # 23
        "No_Beard",           # 24
        "Oval_Face",          # 25
        "Pale_Skin",          # 26
        "Pointy_Nose",        # 27
        "Receding_Hairline",  # 28
        "Rosy_Cheeks",        # 29
        "Sideburns",          # 30
        "Smiling",            # 31
        "Straight_Hair",      # 32
        "Wavy_Hair",          # 33
        "Wearing_Earrings",   # 34
        "Wearing_Hat",        # 35
        "Wearing_Lipstick",   # 36
        "Wearing_Necklace",   # 37
        "Wearing_Necktie",    # 38
        "Young",              # 39
    ]     

        DISEN_BASE_IDX = [
        "5_o_Clock_Shadow",   # 0
        "Arched_Eyebrows",    # 1
        "Attractive",         # 2
        "Bags_Under_Eyes",    # 3
        "Bald",               # 4
        "Bangs",              # 5
        "Big_Lips",           # 6
        "Big_Nose",           # 7
        "Black_Hair",         # 8
        "Blond_Hair",         # 9
        "Blurry",             # 10
        "Brown_Hair",         # 11
        "Bushy_Eyebrows",     # 12
        "Chubby",             # 13
        "Double_Chin",        # 14
        "Eyeglasses",         # 15
        "Goatee",             # 16
        "Gray_Hair",          # 17
        "Heavy_Makeup",       # 18
        "High_Cheekbones",    # 19
        "Male",               # 20
        "Mouth_Slightly_Open",# 21
        "Mustache",           # 22
        "Narrow_Eyes",        # 23
        "No_Beard",           # 24
        "Oval_Face",          # 25
        "Pale_Skin",          # 26
        "Pointy_Nose",        # 27
        "Receding_Hairline",  # 28
        "Rosy_Cheeks",        # 29
        "Sideburns",          # 30
        "Smiling",            # 31
        "Straight_Hair",      # 32
        "Wavy_Hair",          # 33
        "Wearing_Earrings",   # 34
        "Wearing_Hat",        # 35
        "Wearing_Lipstick",   # 36
        "Wearing_Necklace",   # 37
        "Wearing_Necktie",    # 38
        "Young",              # 39
        ]

        REPRESENTANT = [
        "Eyeglasses",         # 15
        "Male",               # 20
        "Pale_Skin",          # 26
        "Smiling",            # 31
        "Wearing_Hat",        # 35
        "Wearing_Lipstick",   # 36
    ]
        # fmt: on

        REPRESENTANT_IDX = [15, 20, 26, 31, 35, 36]


@dataclass
class MNIST:
    @dataclass
    class Path:
        data_dir: str = join(DATA_ROOT, "mnist")


@dataclass
class DSprites:
    @dataclass
    class Path:
        ROOT = join(DATA_ROOT, "dsprites")
        H5 = join(ROOT, "dsprites.h5")
        TRAIN = join(ROOT, "dsprites_train.npz")
        VAL = join(ROOT, "dsprites_val.npz")
        TEST = join(ROOT, "dsprites_test.npz")
        NPZ = join(ROOT, "dsprites.npz")
        BUFF_IMG = join(ROOT, "images_train_buff.pt")
        BUFF_LABELS = join(ROOT, "labels_train_buff.pt")

    @dataclass
    class Params:
        FACTORS_IN_ORDER = ["shape", "scale", "orientation", "pos_x", "pos_y"]

        NUM_VALUES_PER_FACTOR = {
            "shape": 3,
            "scale": 6,
            "orientation": 40,
            "pos_x": 32,
            "pos_y": 32,
        }


@dataclass
class Cars3D:
    @dataclass
    class Path:
        ROOT = join(DATA_ROOT, "cars3d")
        CACHE = join(ROOT, "cache")
        LOCAL = ROOT
        TRAIN = join(ROOT, "cars3d_train.npz")
        VAL = join(ROOT, "cars3d_val.npz")
        TEST = join(ROOT, "cars3d_test.npz")

    @dataclass
    class Params:
        FACTORS_IN_ORDER = ["identity", "elevation_angle", "azimuth_angle"]

        NUM_VALUES_PER_FACTOR = {
            "identity": 183,
            "elevation_angle": 4,
            "azimuth_angle": 24,
        }
