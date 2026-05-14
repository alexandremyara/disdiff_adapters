from torchvision.datasets import CelebA

__all__ = ["CelebADataset"]


class CelebADataset(CelebA):
    def _check_integrity(self):
        return True
