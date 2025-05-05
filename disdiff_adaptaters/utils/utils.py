import h5py
from typing import Union
import torch
from torch import sort
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def load_h5(h5_path: str) :
    """
    Load datasetH5 object from a H5 file.

    Args:
        h5_path: str path of the h5 file.

    Return:
        list of datasetH5
    """
    dataset_h5 = h5py.File(h5_path, "r")
    return [dataset_h5[key] for key in dataset_h5.keys()]

def split(data, label, ratio: int=0.8) :
    """
    Shuffle and split (data,label) in a train, val and test set.
    The ratio is the size of the train set.
    The val and test are each a half of the remain samples.

    Args:
        data: could be an array, torch.tensor, H5dataset etc...
        label: same type

    Return:
        Tensors for each data/label set.

    """
    print("start split from h5 file")

    idx = torch.randperm(len(data))

    train_idx, _ = sort(idx[:int(len(data)*ratio)])
    
    test_idx, _ = sort(idx[int(len(data)*(ratio)):])

    train_data = data[train_idx]
    train_label = label[train_idx]

    test_data = data[test_idx]
    test_label = label[test_idx]
    print("end split")

    return torch.tensor(train_data), torch.tensor(train_label), torch.tensor(test_data), torch.tensor(test_label)

def collate_images(batch: list):
    """
    ##### From a notebook made by Nicolas Bouriez  (HOW_TO_USE.ipynb - ChAdaViT)


    Collate a batch of images into a list of channels, a list of labels and a mapping of the number of channels per image.

    Args:
        batch (list): A list of tuples of (img, label)

    Return:
        channels_list (torch.Tensor): A tensor of shape (X*num_channels, 1, height, width)
        labels_list (torch.Tensor): A tensor of shape (batch_size, )
        num_channels_list (list): A list of the number of channels per image
    """
    num_channels_list = []
    channels_list = []
    labels_list = [] 

    # Iterate over the list of images and extract the channels
    for image, label in batch:
        labels_list.append(label)
        num_channels = image.shape[0]
        num_channels_list.append(num_channels)

        for channel in range(num_channels):
            channel_image = image[channel, :, :].unsqueeze(0)
            channels_list.append(channel_image)

    channels_list = torch.cat(channels_list, dim=0).unsqueeze(
        1
    )  # Shape: (X*num_channels, 1, height, width)
    
    batched_labels = torch.tensor(labels_list)

    return channels_list, batched_labels, num_channels_list

def display(batch: list) :
    """
    batch_size is always a power of 2.
    """
    nb_samples = len(batch[0]) #batch_size
    nb_col = np.log2(nb_samples)
    nb_row = np.log2(nb_samples)
    fig, axes = plt.subplots(int(nb_row), int(nb_col), figsize=(10,7))

    images, labels = batch

    for i in range(nb_samples):
        ax = axes[int(i//nb_row),int(i%nb_col)]

        img = (images[i]*255).to(torch.uint8)
        ax.imshow(img.permute(1,2,0).numpy())
        ax.set_title(f"{labels[i].numpy()}")

def pca_latent(feats: torch.Tensor, labels: torch.Tensor) :
    pca = PCA(n_components=2)
    latent = []

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10.colors

    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    for mu, logvar in list(zip(*feats)) :
        eps = torch.randn_like(logvar)
        z = mu+torch.exp(0.5*logvar)*eps
        latent.append(z.detach().cpu().numpy())

    latent = np.asarray(latent)
    latent_pca = pca.fit_transform(latent)

    for label in unique_labels:
        idx = labels.squeeze(0) == label
        pts = latent_pca[idx.squeeze(1)]
        plt.scatter(pts[:, 0], pts[:, 1],
                    color=label_to_color[label], label=label, alpha=0.7)
        plt.legend()
    plt.grid()
    plt.show()
