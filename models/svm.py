"""
svm.py — Support Vector Machine Classifier
============================================================
Wraps scikit-learn SVC with the BaseModel interface.
Expects pre-extracted statistical features as input: (N, F).

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from models.base_model import BaseModel


def _cfg(config, key: str, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class SVMModel(BaseModel):
    """
    RBF-kernel SVM with automatic feature scaling.

    Config keys read (from CFG["models"]["architectures"]["svm"])
    --------------------------------------------------------------
    kernel      : str   'rbf' | 'linear' | 'poly'   [rbf]
    C           : float regularisation strength      [1.0]
    gamma       : str   'scale' | 'auto' | float     [scale]
    probability : bool  enable predict_proba         [True]
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)

        arch = {}
        if isinstance(config, dict):
            arch = config.get("models", {}).get("architectures", {}).get("svm", {})

        self.kernel      = arch.get("kernel",      "rbf")
        self.C           = arch.get("C",           1.0)
        self.gamma       = arch.get("gamma",       "scale")
        self.probability = arch.get("probability", True)

        self.scaler = StandardScaler()
        self.model  = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=42,
        )

    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "SVMModel":
        print(f"  [SVM] Fitting on {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features …")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True

        train_acc = (self.model.predict(X_scaled) == y_train).mean()
        print(f"  [SVM] Train accuracy : {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            val_acc = (self.predict(X_val) == y_val).mean()
            print(f"  [SVM] Val   accuracy : {val_acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def get_params(self) -> dict:
        return {"kernel": self.kernel, "C": self.C, "gamma": self.gamma}
