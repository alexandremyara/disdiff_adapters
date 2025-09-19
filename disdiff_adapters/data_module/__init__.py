from .bloodmnist import BloodMNISTDataModule
from .shapes3d import Shapes3DDataModule
from .celeba import CelebADataModule
from .mnist  import MNISTDataModule
from .dsprites import DSpritesDataModule
from .latent import LatentDataModule

__all__=["BloodMNISTDataModule", "Shapes3DDataModule", "CelebADataModule", "MNISTDataModule", "DSpritesDataModule", "LatentDataModule"]