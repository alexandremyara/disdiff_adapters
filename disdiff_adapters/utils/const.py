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
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_train.pt")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_val.pt")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_test.pt")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/split_3dshapes.npz")
        VAE = join(LOG_DIR, "vae/shapes/")
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
        DATA = "/projects/compures/alexandre/disdiff_adapters/PyTorch-VAE/Data/"