"""
lda.py — Linear Discriminant Analysis Classifier
============================================================
Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif

from models.base_model import BaseModel


class LDAModel(BaseModel):
    """
    Linear Discriminant Analysis with optional shrinkage regularisation
    and MI-based feature selection.

    Config keys  (CFG["models"]["architectures"]["lda"])
    -----------------------------------------------------
    solver        : str    'svd' | 'lsqr' | 'eigen'       ['lsqr']
    shrinkage     : float | str | None  'auto' or 0–1      ['auto']
    n_features_mi : int | None  top-K features by MI;
                    None = use all features                 [None]
    tune_shrinkage: bool   grid-search shrinkage on val set [True]
    shrinkage_grid: list   values to search over
                    (only used when tune_shrinkage=True)
    """

    # Grid searched when tune_shrinkage=True and no val set is given
    _DEFAULT_SH_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, "auto"]

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)

        arch = {}
        if isinstance(config, dict):
            arch = config.get("models", {}).get("architectures", {}).get("lda", {})

        self.solver         = arch.get("solver",         "lsqr")
        self.shrinkage      = arch.get("shrinkage",      "auto")
        self.n_features_mi  = arch.get("n_features_mi",  None)
        self.tune_shrinkage = arch.get("tune_shrinkage",  True)
        self.shrinkage_grid = arch.get("shrinkage_grid",  self._DEFAULT_SH_GRID)

        # set during fit — may differ from self.shrinkage after tuning
        self._best_shrinkage = self.shrinkage
        self._feature_idx: Optional[np.ndarray] = None
        self.model: Optional[LinearDiscriminantAnalysis] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute MI and return indices of top-K features."""
        K = self.n_features_mi
        if K is None or K >= X.shape[1]:
            return np.arange(X.shape[1])
        sub_size = min(2000, X.shape[0])
        sub_idx  = np.random.RandomState(42).choice(X.shape[0], sub_size, replace=False)
        mi = mutual_info_classif(X[sub_idx], y[sub_idx], random_state=42, n_neighbors=5)
        return np.argsort(-mi)[:K]

    def _build(self, shrinkage) -> LinearDiscriminantAnalysis:
        """Instantiate an LDA with the given shrinkage."""
        # 'svd' solver doesn't support shrinkage
        solver = self.solver if shrinkage is None else "lsqr"
        return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def _tune(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> object:
        """Return the shrinkage value that maximises val accuracy."""
        best_sh, best_acc = self.shrinkage, -1.0
        for sh in self.shrinkage_grid:
            try:
                clf = self._build(sh)
                clf.fit(X_tr, y_tr)
                acc = float((clf.predict(X_val) == y_val).mean())
                if acc > best_acc:
                    best_acc, best_sh = acc, sh
            except Exception:
                continue
        print(f"  [LDA] Shrinkage tuning → best={best_sh}  val_acc={best_acc:.4f}")
        return best_sh

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LDAModel":

        from collections import Counter
        dist = Counter(y_train.tolist())
        print(f"  [LDA] Fitting on {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features …")
        print(f"  [LDA] Class distribution : {dict(dist)}")

        # ---- MI feature selection ----------------------------------------
        self._feature_idx = self._select_features(X_train, y_train)
        n_sel = len(self._feature_idx)
        if n_sel < X_train.shape[1]:
            print(f"  [LDA] MI feature selection : kept {n_sel} / {X_train.shape[1]}")

        X_tr_sel = X_train[:, self._feature_idx]

        # ---- Shrinkage tuning --------------------------------------------
        if self.tune_shrinkage and X_val is not None and y_val is not None:
            X_val_sel = X_val[:, self._feature_idx]
            self._best_shrinkage = self._tune(X_tr_sel, y_train, X_val_sel, y_val)
        else:
            self._best_shrinkage = self.shrinkage
            print(f"  [LDA] Using shrinkage={self._best_shrinkage} (no tuning)")

        # ---- Final fit on all train data ---------------------------------
        self.model = self._build(self._best_shrinkage)
        self.model.fit(X_tr_sel, y_train)
        self.is_fitted = True

        train_acc = float((self.model.predict(X_tr_sel) == y_train).mean())
        print(f"  [LDA] Train accuracy : {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            val_acc = float((self.predict(X_val) == y_val).mean())
            print(f"  [LDA] Val   accuracy : {val_acc:.4f}")

        return self

    def _apply_feature_idx(self, X: np.ndarray) -> np.ndarray:
        if self._feature_idx is not None:
            return X[:, self._feature_idx]
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._apply_feature_idx(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._apply_feature_idx(X))

    def get_params(self) -> dict:
        return {
            "solver":        self.solver,
            "shrinkage":     self._best_shrinkage,
            "n_features_mi": self.n_features_mi,
        }
