from os.path import join
from os import getenv
from dataclasses import dataclass
from typing import ClassVar

#General Const
PROJECT_PATH= "."

LOG_DIR = join(PROJECT_PATH, "disdiff_adapters/logs")

#Const by Dataset/Model
@dataclass
class Shapes3D :

    @dataclass
    class Path :
        H5 = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/3dshapes.h5")
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_train.npz")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_val.npz")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_test.npz")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/split_3dshapes.npz")
        BUFF_IMG = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/images_train_buff.pt")
        BUFF_LABELS = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/labels_train_buff.pt")
    @dataclass
    class Params :
        FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                        'orientation']
        
        NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                            'scale': 8, 'shape': 4, 'orientation': 15}

@dataclass
class ChAda :

    @dataclass
    class Path :
        WEIGHTS = join(PROJECT_PATH, "disdiff_adapters/arch/chada/weights.ckpt")

    @dataclass
    class Config :
        BASE: ClassVar[dict[str, int|bool]] = {"patch_size": 16, "embed_dim": 192, "return_all_tokens": False, "max_number_channels": 10}

@dataclass
class BloodMNIST :
    
    @dataclass
    class Path :
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/bloodmnist/bloodmnist_train.pt")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/bloodmnist/bloodmnist_val.pt")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/bloodmnist/bloodmnist_test.pt")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/bloodmnist/bloodmnist.npz")
        H5 = join(PROJECT_PATH,"disdiff_adapters/data/bloodmnist/bloodmnist.h5" )
        VAE = join(LOG_DIR, "vae/bloodmnist")

    # @dataclass
    # class Params :
    #     N_TRAIN = 
    #     N_VAL =
    #     N_TEST = 

@dataclass
class CelebA :

    @dataclass
    class Path :
        DATA = "/projects/compures/alexandre/PyTorch-VAE/Data/"

    class Params : 
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

@dataclass
class MNIST:
    @dataclass
    class Path :
        data_dir: str = join(PROJECT_PATH, "disdiff_adapters/data/mnist")

@dataclass
class DSprites:
    @dataclass
    class Path :
        H5 = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites.h5")
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites_train.npz")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites_val.npz")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites_test.npz")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites.npz")
        BUFF_IMG = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/images_train_buff.pt")
        BUFF_LABELS = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/labels_train_buff.pt")

    @dataclass
    class Params :
        FACTORS_IN_ORDER = ['color', 'shape', 'scale', 'orientation', 'pos_x',
                        'pos_y']
        
        NUM_VALUES_PER_FACTOR = {'color': 1, 'shape': 3, 'scale': 6, 
                            'orientation': 40, 'pos_x': 32, 'pos_y': 32}