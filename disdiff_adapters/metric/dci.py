"""
DCI (Disentanglement, Completeness, Informativeness) metric.
Ported from notebook/metric.ipynb DCIscore class (### DCI / #### Automatisation).
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


class DCIscore:
    """
    DCI score computation ported from notebook/metric.ipynb.
    Accepts raw tensors (z_s, z_t, labels) directly instead of loading from checkpoint.
    
    Uses val data to fit regressors and test data to compute scores.
    """

    def __init__(
        self,
        z_s_tr: torch.Tensor,
        z_t_tr: torch.Tensor,
        label_tr: torch.Tensor,
        z_s_te: torch.Tensor,
        z_t_te: torch.Tensor,
        label_te: torch.Tensor,
        n_samples_tr: int | None = None,
        n_samples_te: int | None = None,
        verbose: bool = False,
    ):
        self.verbose = verbose

        # Concatenate and normalize train data (same as notebook)
        z_tr = torch.cat([z_s_tr, z_t_tr], dim=1).cpu().numpy()
        self.z_tr = (z_tr - z_tr.mean(axis=0, keepdims=True)) / (
            z_tr.std(axis=0, keepdims=True) + 1e-8
        )
        self.y_tr = label_tr.cpu().numpy().astype(np.int64)
        self.n_samples_tr = self.y_tr.shape[0] if n_samples_te is None else n_samples_tr

        # Concatenate and normalize test data (same as notebook)
        z_te = torch.cat([z_s_te, z_t_te], dim=1).cpu().numpy()
        self.z_te = (z_te - z_te.mean(axis=0, keepdims=True)) / (
            z_te.std(axis=0, keepdims=True) + 1e-8
        )
        self.y_te = label_te.cpu().numpy().astype(np.int64)
        self.n_samples_te = self.y_te.shape[0] if n_samples_te is None else n_samples_te

        self.regressors = None
        self.P_d_given_k = None
        self.P_k_given_d = None

    def train_reg(self):
        """Train a RandomForestRegressor for each factor."""
        regressors = {str(k): {} for k in range(self.y_tr.shape[1])}
        iterator = range(self.y_tr.shape[1])
        if self.verbose:
            iterator = tqdm(iterator, desc="Training DCI regressors")

        for k in iterator:
            # Use torch.randperm like notebook
            perm = torch.randperm(len(self.z_tr))
            reg_k = RandomForestRegressor(
                n_estimators=20,
                max_depth=20,
                n_jobs=-1,
            )
            reg_k.fit(
                self.z_tr[perm][: self.n_samples_tr],
                self.y_tr[:, k][perm][: self.n_samples_tr],
            )

            perm = torch.randperm(len(self.z_tr))
            score_tr = reg_k.score(
                self.z_tr[perm][: self.n_samples_tr],
                self.y_tr[:, k][perm][: self.n_samples_tr],
            )
            perm = torch.randperm(len(self.z_te))
            score_te = reg_k.score(
                self.z_te[perm][: self.n_samples_te],
                self.y_te[:, k][perm][: self.n_samples_te],
            )

            if self.verbose:
                print(f"Reg_{k} score={score_tr}, {score_te}")
            regressors[str(k)]["model"] = reg_k
            regressors[str(k)]["score_tr"] = score_tr
            regressors[str(k)]["score_te"] = score_te
        self.regressors = regressors

    def compute_weights(self):
        """Compute feature importance weights from trained regressors."""
        # 1) Récupérer la matrice d'importances R (D, K) à partir de tes régressions
        D = self.z_tr.shape[1]
        K = self.y_tr.shape[1]

        R = np.zeros((D, K), dtype=float)  # feature importances pour chaque facteur k
        for k in range(K):
            model = self.regressors[str(k)]["model"]
            imp = getattr(model, "feature_importances_", None)
            if imp is None:
                raise ValueError(f"Aucune feature_importances_ pour k={k}")
            if len(imp) != D:
                raise ValueError(f"Dim mismatch: len(imp)={len(imp)} vs D={D}")
            R[:, k] = imp

        # 2) Normaliser par colonne -> P(d | k): "où vit le facteur k ?"
        col_sum = R.sum(axis=0, keepdims=True)  # (1, K)
        col_sum[col_sum == 0] = 1.0  # éviter /0 si colonne nulle
        self.P_d_given_k = R / col_sum  # (D, K)

        # 3) (optionnel) Normaliser par ligne -> P(k | d): "quel facteur porte la dimension d ?"
        row_sum = R.sum(axis=1, keepdims=True)  # (D, 1)
        row_sum[row_sum == 0] = 1.0
        self.P_k_given_d = R / row_sum  # (D, K)

    def dci_scores(self, eps=1e-12):
        """
        P_kd : array (K, D) d'importances non-négatives (p.ex. permutation importance, gain, ou P(d|k)).
            Pas besoin d'être normalisé: on renormalisera correctement pour D et C.
        r2_per_factor : iterable de longueur K avec les R^2 (test) pour l'informativeness (optionnel).

        Retourne: D, C, I (I peut être None si r2_per_factor est None).
        """
        r2_per_factor = [reg["score_te"] for reg in self.regressors.values()]

        # self.P_d_given_k est (D, K) => on transpose pour avoir (K, D)
        P_kd = self.P_d_given_k.T  # (K, D)
        R = np.asarray(P_kd, dtype=float)
        K, D = R.shape

        R = np.clip(R, 0.0, None)
        if R.sum() == 0:
            raise ValueError("La matrice d'importances est nulle.")

        # Poids des dimensions et des facteurs (somme des importances)
        w_d = R.sum(axis=0)  # (D,)
        w_k = R.sum(axis=1)  # (K,)
        w_d = w_d / (w_d.sum() + eps)
        w_k = w_k / (w_k.sum() + eps)

        # p(k|d) : normalisation par colonnes ; p(d|k) : normalisation par lignes
        P_k_given_d = R / (R.sum(axis=0, keepdims=True) + eps)  # (K, D)
        P_d_given_k = R / (R.sum(axis=1, keepdims=True) + eps)  # (K, D)

        # Entropies
        def entropy(p, axis):
            p = np.clip(p, eps, 1.0)
            return -(p * np.log(p)).sum(axis=axis)

        H_k_given_d = entropy(P_k_given_d, axis=0)  # (D,)
        H_d_given_k = entropy(P_d_given_k, axis=1)  # (K,)

        # Disentanglement et completeness (entropies normalisées)
        D_score = float(((1.0 - H_k_given_d / (np.log(K) + eps)) * w_d).sum())
        C_score = float(((1.0 - H_d_given_k / (np.log(D) + eps)) * w_k).sum())

        # Informativeness = moyenne des R^2 test (tronqués à [0,1])
        I_score = None
        if r2_per_factor is not None:
            r2 = np.asarray(r2_per_factor, dtype=float)
            I_score = float(np.clip(r2, 0.0, 1.0).mean())

        return D_score, C_score, I_score

    def compute(self):
        """
        Run the full DCI computation pipeline.

        Returns:
            (D, C, I): disentanglement, completeness, informativeness scores
        """
        self.train_reg()
        self.compute_weights()
        D, C, I = self.dci_scores()
        if self.verbose:
            print(f"D:{D}")
            print(f"C={C}")
            print(f"I={I}")
        return D, C, I
