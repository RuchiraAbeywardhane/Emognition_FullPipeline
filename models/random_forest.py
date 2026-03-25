"""
random_forest.py — Random Forest Classifier
============================================================
Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest with no feature scaling required.

    Config keys  (CFG["models"]["architectures"]["random_forest"])
    --------------------------------------------------------------
    n_estimators : int   number of trees          [200]
    max_depth    : int   max tree depth           [None]
    n_jobs       : int   parallel jobs (-1=all)   [-1]
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)

        arch = {}
        if isinstance(config, dict):
            arch = config.get("models", {}).get("architectures", {}).get("random_forest", {})

        self.n_estimators = arch.get("n_estimators", 200)
        self.max_depth    = arch.get("max_depth",    None)
        self.n_jobs       = arch.get("n_jobs",       -1)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=42,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "RandomForestModel":
        print(f"  [RandomForest] Fitting on {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features …")
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        train_acc = (self.model.predict(X_train) == y_train).mean()
        print(f"  [RandomForest] Train accuracy : {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            val_acc = (self.predict(X_val) == y_val).mean()
            print(f"  [RandomForest] Val   accuracy : {val_acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_params(self) -> dict:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth}
