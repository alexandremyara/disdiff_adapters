"""
Lightning callbacks for computing disentanglement metrics during training.
Implements FactorVAE and DCI metrics ported from notebook/metric.ipynb.
"""

from collections import Counter, defaultdict

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info
from tqdm import tqdm


class DisentanglementMetricsCallback(Callback):
    """
    Callback that computes FactorVAE and DCI metrics after each training epoch.

    Uses the already-buffered latents from XFactors model (latent_val_buff, labels_val_buff)
    and the training buffers (latent_train_buff, labels_train_buff) for DCI train/test split.

    Metrics are logged to all configured loggers (TensorBoard, WandB, etc.).

    Args:
        compute_every_n_epochs: Frequency of metric computation (default: 5)
        n_iter: Number of iterations for FactorVAE metric (default: 153600)
        batch_size: Batch size for FactorVAE sampling (default: 64)
        verbose: Whether to print progress (default: False)
    """

    def __init__(
        self,
        compute_every_n_epochs: int = 5,
        n_iter: int = 153600,
        batch_size: int = 64,
        verbose: bool = False,
    ):
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.rng = np.random.default_rng(0)

    def on_train_epoch_end(self, trainer, pl_module):
        """Compute and log disentanglement metrics at end of training epoch.

        This runs AFTER validation and AFTER model's on_validation_epoch_end,
        ensuring buffers have been concatenated.
        """
        # Skip during sanity check
        if getattr(trainer, "sanity_checking", False):
            return
        # Only compute on main process
        if pl_module.global_rank != 0:
            return
        # Frequency check
        if trainer.current_epoch % self.compute_every_n_epochs != 0:
            return

        # Validate buffers - raises if not ready
        self._validate_buffers(pl_module)

        # Get latents (already concatenated in on_validation_epoch_end of xfactors)
        z_s_val = pl_module.latent_val_buff["s"]
        z_t_val = pl_module.latent_val_buff["t"]
        labels_val = pl_module.labels_val_buff

        z_s_train = pl_module.latent_train_buff["s"]
        z_t_train = pl_module.latent_train_buff["t"]
        labels_train = pl_module.labels_train_buff

        # Get model config
        factor_names = pl_module.hparams.map_idx_labels

        # Compute FactorVAE score
        factorvae_score = self._compute_factorvae(
            z_s_train=z_s_train,
            z_t_train=z_t_train,
            labels_train=labels_train,
            z_s_test=z_s_val,
            z_t_test=z_t_val,
            labels_test=labels_val,
            factor_names=factor_names,
        )
        pl_module.log("metric/factorvae", factorvae_score, sync_dist=True)

        # Compute DCI scores
        d_score, c_score, i_score = self._compute_dci(
            z_s_train=z_s_train,
            z_t_train=z_t_train,
            labels_train=labels_train,
            z_s_test=z_s_val,
            z_t_test=z_t_val,
            labels_test=labels_val,
        )
        pl_module.log("metric/dci_d", d_score, sync_dist=True)
        pl_module.log("metric/dci_c", c_score, sync_dist=True)
        pl_module.log("metric/dci_i", i_score, sync_dist=True)

        rank_zero_info(
            f"[Epoch {trainer.current_epoch}] FactorVAE={factorvae_score:.4f}, "
            f"DCI: D={d_score:.4f}, C={c_score:.4f}, I={i_score:.4f}"
        )

    def _validate_buffers(self, pl_module) -> None:
        """Validate that all required buffers are available and populated. Raises on failure."""
        # Check buffer attributes exist
        if not hasattr(pl_module, "latent_val_buff"):
            raise RuntimeError(
                "DisentanglementMetrics: pl_module.latent_val_buff not found"
            )
        if not hasattr(pl_module, "labels_val_buff"):
            raise RuntimeError(
                "DisentanglementMetrics: pl_module.labels_val_buff not found"
            )
        if not hasattr(pl_module, "latent_train_buff"):
            raise RuntimeError(
                "DisentanglementMetrics: pl_module.latent_train_buff not found"
            )
        if not hasattr(pl_module, "labels_train_buff"):
            raise RuntimeError(
                "DisentanglementMetrics: pl_module.labels_train_buff not found"
            )

        # Get buffer contents
        z_s_val = pl_module.latent_val_buff.get("s")
        z_t_val = pl_module.latent_val_buff.get("t")
        labels_val = pl_module.labels_val_buff

        z_s_train = pl_module.latent_train_buff.get("s")
        z_t_train = pl_module.latent_train_buff.get("t")
        labels_train = pl_module.labels_train_buff

        # Get dimension info
        dim_s = pl_module.hparams.latent_dim_s
        dim_t = sum(pl_module.hparams.dims_by_factors)

        # Validate val buffers are tensors
        if not isinstance(z_s_val, torch.Tensor):
            raise RuntimeError(
                f"DisentanglementMetrics: latent_val_buff['s'] is {type(z_s_val)}, expected Tensor. "
                f"Callback may be running before model's on_validation_epoch_end concatenates buffers."
            )
        # Only check numel > 0 if dim_s > 0
        if dim_s > 0 and z_s_val.numel() == 0:
            raise RuntimeError(
                "DisentanglementMetrics: latent_val_buff['s'] is empty but dim_s > 0"
            )

        if not isinstance(z_t_val, torch.Tensor):
            raise RuntimeError(
                f"DisentanglementMetrics: latent_val_buff['t'] is {type(z_t_val)}, expected Tensor"
            )
        # Only check numel > 0 if dim_t > 0
        if dim_t > 0 and z_t_val.numel() == 0:
            raise RuntimeError(
                "DisentanglementMetrics: latent_val_buff['t'] is empty but dim_t > 0"
            )

        if not isinstance(labels_val, torch.Tensor):
            raise RuntimeError(
                f"DisentanglementMetrics: labels_val_buff is {type(labels_val)}, expected Tensor"
            )
        if labels_val.numel() == 0:
            raise RuntimeError("DisentanglementMetrics: labels_val_buff is empty")

        # Validate train buffers are tensors
        if not isinstance(z_s_train, torch.Tensor):
            raise RuntimeError(
                f"DisentanglementMetrics: latent_train_buff['s'] is {type(z_s_train)}, expected Tensor"
            )
        if dim_s > 0 and z_s_train.numel() == 0:
            raise RuntimeError(
                "DisentanglementMetrics: latent_train_buff['s'] is empty but dim_s > 0"
            )

        if not isinstance(z_t_train, torch.Tensor):
            raise RuntimeError(
                f"DisentanglementMetrics: latent_train_buff['t'] is {type(z_t_train)}, expected Tensor"
            )
        if dim_t > 0 and z_t_train.numel() == 0:
            raise RuntimeError(
                "DisentanglementMetrics: latent_train_buff['t'] is empty but dim_t > 0"
            )

        if not isinstance(labels_train, torch.Tensor):
            raise RuntimeError(
                f"DisentanglementMetrics: labels_train_buff is {type(labels_train)}, expected Tensor"
            )
        if labels_train.numel() == 0:
            raise RuntimeError("DisentanglementMetrics: labels_train_buff is empty")

    # ========== FactorVAE Score ==========
    # Ported exactly from notebook/metric.ipynb FactorVAEScore class

    def _value_index(self, ys: np.ndarray) -> list[dict]:
        """Build factor->value->indices mapping."""
        out = []
        for k in range(ys.shape[0]):
            d = {}
            for v in np.unique(ys[k]):
                d[int(v)] = np.flatnonzero(ys[k] == v)
            out.append(d)
        return out

    def _collect(
        self, mus: np.ndarray, ys: np.ndarray, n_iter: int, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Core FactorVAE collection loop.
        For each iteration:
          1. Pick a random factor k
          2. Pick a random value v for that factor
          3. Sample batch_size points with factor k = v
          4. Find the latent dimension with minimum variance -> argmin
        """
        z_std = mus.std(axis=1, keepdims=True)
        z_std[z_std == 0] = 1.0
        v2i = self._value_index(ys)
        argmins, labels = [], []

        iterator = range(n_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="FactorVAE metric")

        for _ in iterator:
            k = self.rng.integers(0, ys.shape[0])  # Choose a factor f_k
            v = self.rng.choice(list(v2i[k].keys()))  # Choose a value for f_k
            pool = v2i[k][v]
            idx = self.rng.choice(
                pool, size=batch_size, replace=(len(pool) < batch_size)
            )  # Batch with f_k=v

            Z = mus[:, idx] / z_std
            d = int(Z.var(axis=1).argmin())  # get the argmin variance for this batch
            argmins.append(d)
            labels.append(k)

        return np.array(argmins), np.array(labels)

    def _compute_factorvae(
        self,
        z_s_train: torch.Tensor,
        z_t_train: torch.Tensor,
        labels_train: torch.Tensor,
        z_s_test: torch.Tensor,
        z_t_test: torch.Tensor,
        labels_test: torch.Tensor,
        factor_names: list[str],
    ) -> float:
        """
        Compute FactorVAE score using the methodology from metric.ipynb.

        Uses train/val split for building dim->factor mapping (via majority vote on val)
        and computing accuracy on test.
        """
        # Format validation data (for building the mapping)
        Z_val = torch.cat([z_s_train, z_t_train], dim=1).cpu().numpy()
        Z_val = (Z_val - Z_val.mean(axis=0, keepdims=True)) / (
            Z_val.std(axis=0, keepdims=True) + 1e-8
        )
        Y_val = labels_train.cpu().numpy().astype(np.int64)
        mus_val = Z_val.T
        ys_val = Y_val.T

        # Format test data
        Z_test = torch.cat([z_s_test, z_t_test], dim=1).cpu().numpy()
        Z_test = (Z_test - Z_test.mean(axis=0, keepdims=True)) / (
            Z_test.std(axis=0, keepdims=True) + 1e-8
        )
        Y_test = labels_test.cpu().numpy().astype(np.int64)
        mus_test = Z_test.T
        ys_test = Y_test.T

        # Step 1: Build dim->factor mapping using validation data
        argmins_val, labels_val_arr = self._collect(
            mus_val, ys_val, self.n_iter, self.batch_size
        )
        dim_factor_score = self._build_dim_factor_score(
            argmins_val, labels_val_arr, factor_names
        )
        map_dim_factor = self._compute_map(dim_factor_score, factor_names)

        # Step 2: Compute score on test data
        argmins_test, labels_test_arr = self._collect(
            mus_test, ys_test, self.n_iter, self.batch_size
        )

        predictions = []
        for argmin in argmins_test:
            pred_str = map_dim_factor.get(str(argmin), "s")
            if pred_str == "s":
                pred_int = -1
            else:
                try:
                    pred_int = factor_names.index(pred_str)
                except ValueError:
                    pred_int = -1
            predictions.append(pred_int)

        predictions = np.asarray(predictions)
        score = np.sum(predictions == labels_test_arr) / self.n_iter

        return float(score)

    def _build_dim_factor_score(
        self, argmins: np.ndarray, labels: np.ndarray, factor_names: list[str]
    ) -> dict:
        """Build dim->factor association scores from collected argmins/labels."""
        dim_factor_score = defaultdict(lambda: defaultdict(float))

        for d in np.unique(argmins):
            cnt = Counter(labels[argmins == d])
            total = sum(cnt.values())
            for k, n in cnt.most_common():
                dim_factor_score[str(d)][factor_names[k]] = n / total

        return dict(dim_factor_score)

    def _compute_map(
        self, dim_factor_score: dict, factor_names: list[str]
    ) -> dict[str, str]:
        """Compute dim->factor mapping using majority vote."""
        map_dim_factor = {}
        max_dim = (
            max(int(k) for k in dim_factor_score.keys()) if dim_factor_score else 0
        )

        for dim in range(max_dim + 1):
            factors = dim_factor_score.get(str(dim), {})
            if not factors or isinstance(factors, list):
                first_factor = "s"
            else:
                first_factor = list(factors.keys())[0] if factors else "s"
            map_dim_factor[str(dim)] = first_factor

        return map_dim_factor

    # ========== DCI Score ==========
    # Ported from notebook/metric.ipynb DCIscore class

    def _compute_dci(
        self,
        z_s_train: torch.Tensor,
        z_t_train: torch.Tensor,
        labels_train: torch.Tensor,
        z_s_test: torch.Tensor,
        z_t_test: torch.Tensor,
        labels_test: torch.Tensor,
    ) -> tuple[float, float, float]:
        """Compute DCI scores using the DCIscore class."""
        from disdiff_adapters.metric.dci import DCIscore

        dci = DCIscore(
            z_s_tr=z_s_train,
            z_t_tr=z_t_train,
            label_tr=labels_train,
            z_s_te=z_s_test,
            z_t_te=z_t_test,
            label_te=labels_test,
            verbose=self.verbose,
        )
        return dci.compute()
