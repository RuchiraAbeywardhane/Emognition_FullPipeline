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

from models.base_model import BaseModel


class LDAModel(BaseModel):
    """
    Linear Discriminant Analysis.
    No hyper-parameters needed — LDA is fully determined by the data.
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)
        self.model = LinearDiscriminantAnalysis()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LDAModel":
        print(f"  [LDA] Fitting on {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features …")
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        train_acc = (self.model.predict(X_train) == y_train).mean()
        print(f"  [LDA] Train accuracy : {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            val_acc = (self.predict(X_val) == y_val).mean()
            print(f"  [LDA] Val   accuracy : {val_acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_params(self) -> dict:
        return {}
