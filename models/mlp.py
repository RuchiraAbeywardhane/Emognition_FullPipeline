"""
mlp.py — Multi-Layer Perceptron Classifier
============================================================
Operates on pre-extracted feature vectors: (N, F).
First deep model in the pipeline — a simple fully-connected network.

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from models.base_model import BaseModel


# ===========================================================================
# PyTorch Module
# ===========================================================================

class _MLPNet(nn.Module):
    """
    Fully-connected network.
    Input  : (batch, F)
    Output : (batch, num_classes)  — raw logits
    """

    def __init__(self, in_features: int, hidden_layers: List[int],
                 num_classes: int, dropout: float, activation: str):
        super().__init__()

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(
            activation.lower(), nn.ReLU
        )

        layers = []
        prev = in_features
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), act_fn(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ===========================================================================
# BaseModel wrapper
# ===========================================================================

def _compute_class_weights(y: np.ndarray, num_classes: int) -> "torch.Tensor":
    """Compute inverse-frequency class weights and return as a CPU tensor."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)          # avoid /0
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes    # normalise
    return torch.tensor(weights, dtype=torch.float32)


class MLPModel(BaseModel):
    """
    MLP classifier with early stopping.

    Config keys  (CFG["models"]["architectures"]["mlp"])
    ----------------------------------------------------
    hidden_layers : list of int   neuron counts per layer   [[256,128,64]]
    dropout       : float         dropout probability       [0.3]
    activation    : str           'relu' | 'gelu' | 'tanh' ['relu']

    Training config  (CFG["training"])
    -----------------------------------
    epochs        : int    [50]
    batch_size    : int    [32]
    learning_rate : float  [1e-3]
    weight_decay  : float  [1e-4]
    patience      : int    early-stopping patience  [10]
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLPModel. pip install torch")

        arch     = {}
        training = {}
        if isinstance(config, dict):
            arch     = config.get("models", {}).get("architectures", {}).get("mlp", {})
            training = config.get("training", {})

        self.hidden_layers = arch.get("hidden_layers", [256, 128, 64])
        self.dropout       = arch.get("dropout",       0.3)
        self.activation    = arch.get("activation",    "relu")

        self.epochs        = training.get("epochs",        50)
        self.batch_size    = training.get("batch_size",    32)
        self.lr            = training.get("learning_rate", 1e-3)
        self.weight_decay  = training.get("weight_decay",  1e-4)
        self.patience      = training.get("patience",      10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_: Optional[_MLPNet] = None

    # ------------------------------------------------------------------

    def _build(self, in_features: int) -> None:
        self.net_ = _MLPNet(
            in_features, self.hidden_layers,
            self.num_classes, self.dropout, self.activation,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray,
                     shuffle: bool = True) -> DataLoader:
        tx = torch.tensor(X, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(tx, ty),
                          batch_size=self.batch_size, shuffle=shuffle)

    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "MLPModel":

        from collections import Counter
        dist = Counter(y_train.tolist())

        self._build(X_train.shape[1])
        optimizer = torch.optim.Adam(self.net_.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6)

        class_weights = _compute_class_weights(y_train, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        best_val_loss = float("inf")
        no_improve    = 0

        print(f"  [MLP] Training on {X_train.shape[0]} samples | "
              f"device={self.device} | epochs={self.epochs}")
        print(f"  [MLP] Class distribution : {dict(dist)}")
        print(f"  [MLP] Class weights      : {class_weights.tolist()}")

        for epoch in range(1, self.epochs + 1):
            # --- train ---
            self.net_.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net_(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_loss = total_loss / len(X_train)

            # --- val ---
            if X_val is not None and y_val is not None:
                self.net_.eval()
                with torch.no_grad():
                    xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    yv = torch.tensor(y_val, dtype=torch.long).to(self.device)
                    val_loss = criterion(self.net_(xv), yv).item()
                    val_acc  = (self.net_(xv).argmax(1) == yv).float().mean().item()
                scheduler.step(val_loss)

                if epoch % 10 == 0:
                    print(f"    epoch {epoch:3d}  train_loss={avg_loss:.4f}  "
                          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    no_improve    = 0
                    self._best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        print(f"    Early stopping at epoch {epoch}")
                        break
            else:
                if epoch % 10 == 0:
                    print(f"    epoch {epoch:3d}  train_loss={avg_loss:.4f}")

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.net_.load_state_dict(self._best_state)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.net_(tx).argmax(1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(self.device)
            return torch.softmax(self.net_(tx), dim=1).cpu().numpy()

    def get_params(self) -> dict:
        return {"hidden_layers": self.hidden_layers, "dropout": self.dropout,
                "activation": self.activation, "epochs": self.epochs}
