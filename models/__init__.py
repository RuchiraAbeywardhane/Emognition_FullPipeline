"""
models/__init__.py — Model Registry
============================================================
Central lookup so the trainer can instantiate any model by name.

Usage
-----
    from models import build_model
    from config.config import CFG

    model = build_model("svm", num_classes=4, config=CFG)
    model = build_model("transformer", num_classes=4, config=CFG)

Available models
----------------
  Simple (classical ML — take statistical features (N, F))
    "svm"           SVMModel
    "random_forest" RandomForestModel
    "knn"           KNNModel
    "lda"           LDAModel

  Complex (deep learning — take raw windows (N, W, C))
    "mlp"           MLPModel          (also accepts (N, F))
    "cnn1d"         CNN1DModel
    "lstm"          LSTMModel
    "bilstm"        LSTMModel         (bidirectional=True)
    "cnn_lstm"      CNNLSTMModel
    "transformer"   TransformerModel

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

from models.base_model     import BaseModel
from models.svm            import SVMModel
from models.random_forest  import RandomForestModel
from models.knn            import KNNModel
from models.lda            import LDAModel
from models.mlp            import MLPModel
from models.cnn1d          import CNN1DModel
from models.lstm           import LSTMModel
from models.cnn_lstm       import CNNLSTMModel
from models.transformer    import TransformerModel

# -----------------------------------------------------------------------
# Registry — name → class
# -----------------------------------------------------------------------
_REGISTRY: dict = {
    # Classical ML
    "svm":           SVMModel,
    "random_forest": RandomForestModel,
    "knn":           KNNModel,
    "lda":           LDAModel,
    # Deep learning
    "mlp":           MLPModel,
    "cnn1d":         CNN1DModel,
    "lstm":          LSTMModel,
    "bilstm":        LSTMModel,       # same class, bidirectional flag set via config
    "cnn_lstm":      CNNLSTMModel,
    "transformer":   TransformerModel,
}

# Which models need raw windows (N, W, C) vs feature vectors (N, F)
RAW_WINDOW_MODELS = {"cnn1d", "lstm", "bilstm", "cnn_lstm", "transformer"}
FEATURE_MODELS    = {"svm", "random_forest", "knn", "lda", "mlp"}


def build_model(name: str, num_classes: int, config=None) -> BaseModel:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name        : Model name — one of the keys in _REGISTRY.
    num_classes : Number of output classes.
    config      : CFG dict or None.

    Returns
    -------
    Fitted-ready BaseModel instance.

    Raises
    ------
    ValueError if the name is not recognised.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](num_classes=num_classes, config=config)


def list_models() -> list:
    """Return a sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())


__all__ = [
    "BaseModel",
    "SVMModel", "RandomForestModel", "KNNModel", "LDAModel",
    "MLPModel", "CNN1DModel", "LSTMModel", "CNNLSTMModel", "TransformerModel",
    "build_model", "list_models",
    "RAW_WINDOW_MODELS", "FEATURE_MODELS",
]
