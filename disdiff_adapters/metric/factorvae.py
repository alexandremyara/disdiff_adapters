import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

def compute_factorvae_score(latent_t: torch.Tensor, shapes: torch.Tensor) -> float:
    """
    latent_t: (N, d_t)
    shapes: (N,) valeurs dans [0, 1, 2, 3]
    """
    latent_t = latent_t.detach().cpu().numpy()
    shapes = shapes.detach().cpu().numpy()

    votes = []
    for shape_val in np.unique(shapes):
        idx = np.where(shapes == shape_val)[0]
        if len(idx) < 2:
            continue
        subset = latent_t[idx]
        var = np.var(subset, axis=0)
        j = np.argmin(var)
        votes.append((j, shape_val))

    vote_dict = defaultdict(lambda: np.zeros(4))  # j → [nb 0, nb 1, ..., nb 3]
    for j, shape in votes:
        vote_dict[j][shape] += 1

    preds = []
    targets = []
    for j, shape in votes:
        pred = np.argmax(vote_dict[j])
        preds.append(pred)
        targets.append(shape)

    return accuracy_score(targets, preds)