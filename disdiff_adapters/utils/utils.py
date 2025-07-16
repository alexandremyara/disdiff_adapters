import h5py
import torch
from torch import sort
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns # type: ignore
from disdiff_adapters.loss import *


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

def display_latent(labels: torch.Tensor, 
               mu_logvars: None|tuple[torch.Tensor]=None,
               z: None|torch.Tensor=None,
               title: str="latent space", 
               test: bool=False,) :
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

    #pca 
    latent_pca = z.detach().cpu().numpy()
    explained_axis = [-1, -1]
    if not z.shape[1] in [1,2] : 
        latent_pca = pca.fit_transform(latent_pca)
        explained_axis = pca.explained_variance_ratio_

    #variance explained : if -2, no pca has been run
    explained = np.sum(explained_axis)

    #plot
    for label in np.random.permutation(unique_labels):
        idx = (labels.squeeze(0) == label)
        idx = idx.detach().cpu()
        pts = latent_pca[idx.squeeze(1).cpu()]
            
        if z.shape[1] == 1 : pts_y = torch.zeros_like(pts[:, 0])
        else : pts_y = pts[:, 1]
        plt.scatter(pts[:, 0], pts_y,
                    color=label_to_color[label], label=label, alpha=0.3)
        plt.xlabel(f"{explained_axis[0]}")
        plt.ylabel(f"{explained_axis[1]}")

        plt.legend()
    plt.title(title+f" explained : {explained}")
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
 


def merge_images(save_gen_path, save_gen_s_path, save_gen_t_path):
    images = [Image.open(path) for path in [save_gen_path, save_gen_s_path, save_gen_t_path]]
    labels = ["generation", "gen_s", "gen_t"]
    widths = [img.width for img in images]
    assert all(w == widths[0] for w in widths)

    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    text_height = font_size + 10
    separator_height = 10
    separator = Image.fromarray(np.zeros((separator_height, widths[0], 3), dtype=np.uint8))

    final_parts = []
    for i, (img, label) in enumerate(zip(images, labels)):
        text_img = Image.new("RGB", (widths[0], text_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        text_width = draw.textlength(label, font=font)
        draw.text(((widths[0] - text_width) // 2, 5), label, fill=(255, 255, 255), font=font)
        final_parts.append(text_img)
        final_parts.append(img)
        if i != len(images) - 1:
            final_parts.append(separator)

    total_height = sum(p.height for p in final_parts)
    final_image = Image.new("RGB", (widths[0], total_height))

    y = 0
    for part in final_parts:
        final_image.paste(part, (0, y))
        y += part.height
    
    return final_image


def merge_images_with_black_gap(image_paths, gap=10):
    images = [Image.open(p) for p in image_paths]
    widths = [img.width for img in images]
    if len(set(widths)) != 1:
        raise ValueError("Toutes les images doivent avoir la même largeur")
    W = widths[0]
    separator = Image.new("RGB", (W, gap), color=(0, 0, 0))
    parts = []
    for img in images[:-1]:
        parts.append(img)
        parts.append(separator)
    parts.append(images[-1])
    total_h = sum(p.height for p in parts)
    merged = Image.new("RGB", (W, total_h), color=(0, 0, 0))
    y = 0
    for p in parts:
        merged.paste(p, (0, y))
        y += p.height
    return merged


def log_cross_cov_heatmap(mu_s, logvar_s, mu_t, logvar_t, save_path: str):
    cov_mu = cross_cov(mu_s, mu_t).detach().cpu().numpy()
    assert cov_mu.shape == (mu_s.shape[1], mu_t.shape[1]), "ERROR COV MATRIX SHAPE"
    cov_logvar = cross_cov(logvar_s, logvar_t).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(cov_mu, ax=axes[0], cmap="coolwarm", center=0, cbar=True,)
    axes[0].set_title("cross_cov(mu_s, mu_t)")

    sns.heatmap(cov_logvar, ax=axes[1], cmap="coolwarm", center=0, cbar=True)
    axes[1].set_title("cross_cov(logvar_s, logvar_t)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)