import lightning as L
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
import numpy as np

from os.path import join, exists

from disdiff_adapters.dataset.shapes3d import Shapes3DDataset
from disdiff_adapters.utils.utils import load_h5, split
from disdiff_adapters.utils.const import Shapes3D

class Shapes3DDataModule(LightningDataModule) :
    
    def __init__(self, h5_path: str=Shapes3D.Path.H5, 
                 train_path: str=Shapes3D.Path.TRAIN,
                 val_path: str=Shapes3D.Path.VAL,
                 test_path: str=Shapes3D.Path.TEST,
                 ratio: int=0.8,
                 batch_size: int=8) :
        super().__init__()
        self.h5_path = h5_path
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path
        self.ratio = ratio
        self.batch_size = batch_size

    def prepare_data(self, is_h5=False):

        if not (exists(self.train_path) and exists(self.val_path) and exists(self.test_path)):
            if is_h5 :
                print("h5 file loading.")
                images, labels = load_h5(self.h5_path)
                train_images, train_labels, test_images, test_labels = split(images, labels)
                train_images, train_labels, val_images, val_labels = split(train_images, train_labels)
            else :
                print("npz file loading.")
                data = np.load(Shapes3D.Path.NPZ)
                train_images = data["train_images.npy"]
                train_images = torch.from_numpy(train_images)

                train_labels = data["train_labels.npy"]
                train_labels = torch.from_numpy(train_labels)

                test_images = data["test_images.npy"]
                test_images = torch.from_numpy(test_images)

                test_labels = data["test_labels.npy"]
                test_labels = torch.from_numpy(test_labels)

                train_images, train_labels, val_images, val_labels = split(train_images, train_labels)

            train_images = (train_images.permute(0,3,1,2)/255).to(torch.float32)
            val_images = (val_images.permute(0,3,1,2)/255).to(torch.float32)    
            test_images = (test_images.permute(0,3,1,2)/255).to(torch.float32)

            print("save tensors\n")
            torch.save((train_images, train_labels), self.train_path)
            torch.save((val_images, val_labels), self.val_path)
            torch.save((test_images, test_labels), self.test_path)
            print("tensors saved.")

        else : pass

    def setup(self, stage) :
        if stage in ("fit", None) :
            print("loading of tensors - train")
            train_images, train_labels = torch.load(self.train_path)
            print("loading of tensors - val")
            val_images, val_labels = torch.load(self.val_path)
            print("load dataset - train")
            self.train_dataset = Shapes3DDataset(train_images, train_labels)
            print("load dataset val")
            self.val_dataset = Shapes3DDataset(val_images, val_labels)
        else :
            test_images, test_labels = torch.load(self.test_path)
            self.test_dataset = Shapes3DDataset(test_images, test_labels)
        print("tensors loaded.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)
