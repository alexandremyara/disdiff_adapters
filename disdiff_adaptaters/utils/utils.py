import h5py
from typing import Union
import torch
import numpy as np

def load_h5(h5_path) :
    dataset_h5 = h5py.File(h5_path, "r")
    return [dataset_h5[key] for key in dataset_h5.keys()]

def split(data, label, ratio: int=0.8) :

    idx = torch.randperm(len(data))
    
    train_idx = idx[:int(len(data)*ratio)]
    val_idx = idx[int(len(data)*ratio) : int(len(data)*(ratio+(1-ratio)/2))]
    test_idx = idx[int(len(data)*(ratio+(1-ratio)/2)):]

    train_data = data[train_idx]
    train_label = label[train_idx]

    val_data = data[val_idx]
    val_label = label[val_idx]

    test_data = data[test_idx]
    test_label = label[test_idx]

    return torch.tensor(train_data), torch.tensor(train_label), torch.tensor(val_data), torch.tensor(val_label), torch.tensor(test_data), torch.tensor(test_label)