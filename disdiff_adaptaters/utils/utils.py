import h5py
import torch
from torch import sort
import numpy as np
import matplotlib.pyplot as plt
import math


from sklearn.decomposition import PCA

def load_h5(h5_path: str) :
    """
    Load datasetH5 object from a H5 file.

    Args:
        h5_path: str path of the h5 file.

    Return:
        list of datasetH5
    """
    try : 
        dataset_h5 = h5py.File(h5_path, "r")
        return [dataset_h5[key] for key in dataset_h5.keys()]
    except FileNotFoundError as e : print("WARNING : file not foud.")
    
def split(data: torch.Tensor, label: torch.Tensor, ratio: float=0.8) :
    """
    Shuffle and split (data,label) in a train, val and test set.
    The ratio is the size of the train set.
    The val and test are each a half of the remain samples.

    Args:
        data: could be an array, torch.tensor, H5dataset etc...
        label: same type

    Return:
        Tensors for each data/label set.
        train_data, train_labels, test_data, test_labels

    """
    print("start split")

    idx = torch.randperm(len(data))

    train_idx, _ = sort(idx[:int(len(data)*ratio)])
    
    test_idx, _ = sort(idx[int(len(data)*(ratio)):])

    train_data = data[train_idx]
    train_label = label[train_idx]

    test_data = data[test_idx]
    test_label = label[test_idx]
    print("end split")

    return torch.tensor(train_data), torch.tensor(train_label), torch.tensor(test_data), torch.tensor(test_label)

def collate_images(batch: list[torch.Tensor]):
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

def display(batch: tuple[torch.Tensor]) :
    """
    Display a batch of RGB images. 
    Batch sould be a tuple of two tensors : ([BATCH_SIZE, 3, H, W], [BATCH_SIZE, 1])
    batch_size is always a power of 2.

    Args:
    batch_size: tuple[torch.Tensor], ([BATCH_SIZE, 3, H, W], [BATCH_SIZE, 1])

    """

    nb_samples = len(batch[0]) #batch_size
    # nb_col = np.log2(nb_samples)
    # nb_row = np.log2(nb_samples)


    nb_col = math.ceil(math.sqrt(nb_samples))
    nb_row = math.ceil(nb_samples / nb_col)

    fig, axes = plt.subplots(int(nb_row), int(nb_col), figsize=(10,7))

    images, labels = batch

    for i in range(nb_samples):
        ax = axes[int(i//nb_row),int(i%nb_col)]

        img = images[i]
        img = 255*(img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img.to(torch.uint8)
        
        ax.imshow(img.permute(1,2,0).numpy())
        ax.set_title(f"{labels[i].numpy()}")

def sample_from(mu_logvar: tuple[torch.Tensor], test=False):
    mu, logvar = mu_logvar
    eps = torch.randn_like(logvar)
    
    if test: return mu
    else : return mu + torch.exp(0.5 * logvar) * eps

def pca_latent(labels: torch.Tensor, 
               mu_logvars: None|tuple[torch.Tensor]=None,
               z: None|torch.Tensor=None, 
               test: bool=False) :
    """
    Generate a plot to visualize in 2D the latent space.
    Ensure that if z=None, mu_logvars is not None.
    
    Args:
    feats: tuple[torch.Tensor], ((number_sample,latent_dim), (number_sample,latent_dim))
    labels: torch.Tensor, (number_sample, 1)
    z: None|torch.Tensor, (number_sample, latent_dim). Allows to give directly the latent vector. 
    test: bool, if inference set True.
    """
    assert (z is not None and mu_logvars is None) or (z is None and mu_logvars is not None), "Among z and mu_logvars, one should be at None. Both can't be."

    pca = PCA(n_components=2)
    latent = []

    unique_labels = np.unique(labels.detach().cpu().numpy())
    colors = plt.cm.tab10.colors

    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    if z is None :
        z = sample_from(mu_logvars, test=test)

    latent_pca = pca.fit_transform(z.detach().cpu().numpy())

    for label in unique_labels:
        idx = labels.squeeze(0) == label
        pts = latent_pca[idx.squeeze(1)]
        plt.scatter(pts[:, 0], pts[:, 1],
                    color=label_to_color[label], label=label, alpha=0.7)
        plt.legend()
    plt.grid()
    plt.show()

def set_device(pref_gpu: int=0) -> str :
    """
    Looking for a GPU and display informations if available.

    Args:
        pref_gpu: int, id of the main gpu.
    Return:
        device: str, name of device (cpu or cuda)
    """
    is_gpu = torch.cuda.is_available()

    device = f"cuda:{pref_gpu}" if is_gpu else "cpu"

    if is_gpu :
        print("Nombre de GPU :", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print(f"\n[ GPU {i} ]")
            print("Nom :", torch.cuda.get_device_name(i))
            print("Mémoire totale :", round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2), "Go")
            print("Mémoire utilisée :", round(torch.cuda.memory_allocated(i) / 1e9, 2), "Go")
            print("Mémoire réservée :", round(torch.cuda.memory_reserved(i) / 1e9, 2), "Go")

    print(f"current device is {torch.cuda.current_device()}")
    return device, is_gpu
 