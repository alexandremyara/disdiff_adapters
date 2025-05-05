import os
from dataclasses import dataclass

PROJECT_PATH=os.getenv("IBENS_PROJECT_PATH")
if PROJECT_PATH is None : PROJECT_PATH ="." 

@dataclass
class Shapes3D :

    @dataclass
    class Path :
        H5 = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/3dshapes.h5")
        TRAIN = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/shapes3d_train.pt")
        VAL = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/shapes3d_val.pt")
        TEST = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/shapes3d_test.pt")
        NPZ = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/split_3dshapes.npz")

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
        WEIGHTS = os.path.join(PROJECT_PATH, "disdiff_adaptaters/arch/chada/weights.ckpt")

    @dataclass
    class Config :
        BASE = {"patch_size": 16, "embed_dim": 192, "return_all_tokens": False, "max_number_channels": 10}

@dataclass
class BloodMNIST :
    
    @dataclass
    class Path :
        TRAIN = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/bloodmnist/bloodmnist_train.pt")
        VAL = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/bloodmnist/bloodmnist_val.pt")
        TEST = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/bloodmnist/bloodmnist_test.pt")
        NPZ = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/bloodmnist/bloodmnist.npz")
        H5 = os.path.join(PROJECT_PATH,"disdiff_adaptaters/data/bloodmnist/bloodmnist.h5" )
