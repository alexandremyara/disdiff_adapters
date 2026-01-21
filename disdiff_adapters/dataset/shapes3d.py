import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["Shapes3DDataset"]


class Shapes3DDataset(Dataset):
    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        self._npz = None
        with np.load(npz_path, mmap_mode="r") as f:
            self.length = f["images"].shape[0]

    def _ensure_loaded(self):
        if self._npz is None:
            self._npz = np.load(self.npz_path, mmap_mode="r")
            self.images = self._npz["images"]
            self.labels = self._npz["labels"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_loaded()
        image = self.images[idx]
        processed_img = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
        label = torch.as_tensor(self.labels[idx])
        return processed_img, label
