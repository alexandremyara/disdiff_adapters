from os.path import join
from os import getenv
from dataclasses import dataclass

#General Const
PROJECT_PATH=getenv("IBENS_PROJECT_PATH")
if PROJECT_PATH is None : PROJECT_PATH ="." 

LOG_DIR = join(PROJECT_PATH, "disdiff_adaptaters/logs")

#Const by Dataset/Model
@dataclass
class Shapes3D :

    @dataclass
    class Path :
        H5 = join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/3dshapes.h5")
        TRAIN = join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/shapes3d_train.pt")
        VAL = join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/shapes3d_val.pt")
        TEST = join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/shapes3d_test.pt")
        NPZ = join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/split_3dshapes.npz")
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
        WEIGHTS = join(PROJECT_PATH, "disdiff_adaptaters/arch/chada/weights.ckpt")

    @dataclass
    class Config :
        BASE = {"patch_size": 16, "embed_dim": 192, "return_all_tokens": False, "max_number_channels": 10}

@dataclass
class BloodMNIST :
    
    @dataclass
    class Path :
        TRAIN = join(PROJECT_PATH, "disdiff_adaptaters/data/bloodmnist/bloodmnist_train.pt")
        VAL = join(PROJECT_PATH, "disdiff_adaptaters/data/bloodmnist/bloodmnist_val.pt")
        TEST = join(PROJECT_PATH, "disdiff_adaptaters/data/bloodmnist/bloodmnist_test.pt")
        NPZ = join(PROJECT_PATH, "disdiff_adaptaters/data/bloodmnist/bloodmnist.npz")
        H5 = join(PROJECT_PATH,"disdiff_adaptaters/data/bloodmnist/bloodmnist.h5" )
        VAE = join(LOG_DIR, "vae/bloodmnist")

@dataclass
class CelebA :

    @dataclass
    class Path :
        DATA = "/projects/compures/alexandre/disdiff_adaptaters/PyTorch-VAE/Data/"