"""
knn.py — K-Nearest Neighbours Classifier
============================================================
Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from models.base_model import BaseModel


class KNNModel(BaseModel):
    """
    K-Nearest Neighbours with feature scaling.

    Config keys  (CFG["models"]["architectures"]["knn"])
    ----------------------------------------------------
    n_neighbors : int   number of neighbours   [5]
    metric      : str   distance metric        ['euclidean']
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)

        arch = {}
        if isinstance(config, dict):
            arch = config.get("models", {}).get("architectures", {}).get("knn", {})

        self.n_neighbors = arch.get("n_neighbors", 5)
        self.metric      = arch.get("metric",      "euclidean")

        self.scaler = StandardScaler()
        self.model  = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "KNNModel":
        print(f"  [KNN] Fitting on {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features …")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True

        train_acc = (self.model.predict(X_scaled) == y_train).mean()
        print(f"  [KNN] Train accuracy : {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            val_acc = (self.predict(X_val) == y_val).mean()
            print(f"  [KNN] Val   accuracy : {val_acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def get_params(self) -> dict:
        return {"n_neighbors": self.n_neighbors, "metric": self.metric}
