import os
from dataclasses import dataclass

PROJECT_PATH="/projects/compures/alexandre/disdiff_adaptaters"

@dataclass
class Shapes3D :

    @dataclass
    class Path :
        H5 = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/3dshapes/3dshapes.h5")
        TRAIN = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/shapes3d_train.pt")
        VAL = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/shapes3d_val.pt")
        TEST = os.path.join(PROJECT_PATH, "disdiff_adaptaters/data/shapes3d_test.pt")

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
        WEIGHTS = os.path.join(PROJECT_PATH, "disdiff_adaptaters/arch/weights.ckpt")