"""
base_model.py — Abstract Base Class for all Emognition models
============================================================
Every model (classical or deep) inherits from BaseModel and
implements the same fit / predict / predict_proba interface so
the trainer and evaluator can treat them identically.

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseModel(ABC):
    """
    Minimal interface shared by every model in the pipeline.

    Classical ML models (SVM, RF, KNN, LDA) wrap a scikit-learn
    estimator.  Deep learning models (MLP, CNN1D, LSTM, …) wrap a
    PyTorch nn.Module and implement their own training loop.

    Parameters
    ----------
    num_classes : Number of output classes.
    config      : Config object or dict — model reads its own hyper-
                  parameters from here.
    """

    def __init__(self, num_classes: int, config=None):
        self.num_classes = num_classes
        self.config      = config
        self.is_fitted   = False

    # ------------------------------------------------------------------
    # Must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "BaseModel":
        """
        Train the model.

        Parameters
        ----------
        X_train : (N, F) feature matrix  or  (N, W, C) raw windows.
        y_train : (N,)   integer labels.
        X_val   : Optional validation features.
        y_val   : Optional validation labels.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : (N, F)  or  (N, W, C)

        Returns
        -------
        (N,) integer predictions.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : (N, F)  or  (N, W, C)

        Returns
        -------
        (N, num_classes) float probabilities.
        """

    # ------------------------------------------------------------------
    # Optional — override for deep models
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return a dict of hyper-parameters (for logging / saving)."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes})"
