"""
lstm.py — Bidirectional LSTM Classifier
============================================================
Operates on raw EEG windows: (N, W, C).
Models temporal dependencies across the signal sequence.

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

from typing import Optional
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

class _LSTMNet(nn.Module):
    """
    Bidirectional LSTM → last hidden state → classifier head.
    Input  : (batch, W, C)   time-first
    Output : (batch, num_classes)
    """

    def __init__(self, n_channels: int, hidden_size: int, num_layers: int,
                 dropout: float, bidirectional: bool, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions   = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        _, (h, _) = self.lstm(x)        # h: (num_layers*dirs, B, hidden)
        # Take last layer's hidden state for both directions
        h = torch.cat([h[-2], h[-1]], dim=-1) if self.lstm.bidirectional \
            else h[-1]                  # (B, hidden * dirs)
        return self.head(self.dropout(h))


# ===========================================================================
# BaseModel wrapper
# ===========================================================================

class LSTMModel(BaseModel):
    """
    Bidirectional LSTM classifier.  Input: raw windows (N, W, C).

    Config keys  (CFG["models"]["architectures"]["lstm"] or "bilstm")
    -----------------------------------------------------------------
    hidden_size   : int   LSTM hidden units         [128]
    num_layers    : int   stacked LSTM layers       [2]
    dropout       : float                           [0.3]
    bidirectional : bool                            [True]

    Training config  (CFG["training"])
    ------------------------------------
    epochs / batch_size / learning_rate / weight_decay / patience
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMModel.")

        arch     = {}
        training = {}
        if isinstance(config, dict):
            # Accept either 'lstm' or 'bilstm' arch key
            archs = config.get("models", {}).get("architectures", {})
            arch  = archs.get("bilstm", archs.get("lstm", {}))
            training = config.get("training", {})

        self.hidden_size   = arch.get("hidden_size",   128)
        self.num_layers    = arch.get("num_layers",    2)
        self.dropout       = arch.get("dropout",       0.3)
        self.bidirectional = arch.get("bidirectional", True)

        self.epochs       = training.get("epochs",        50)
        self.batch_size   = training.get("batch_size",    32)
        self.lr           = training.get("learning_rate", 1e-3)
        self.weight_decay = training.get("weight_decay",  1e-4)
        self.patience     = training.get("patience",      10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_: Optional[_LSTMNet] = None

    def _build(self, n_channels: int) -> None:
        self.net_ = _LSTMNet(
            n_channels, self.hidden_size, self.num_layers,
            self.dropout, self.bidirectional, self.num_classes,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        tx = torch.tensor(X, dtype=torch.float32)          # (N, W, C) — time-first OK
        ty = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(tx, ty),
                          batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LSTMModel":

        N, W, C = X_train.shape
        self._build(C)
        optimizer = torch.optim.Adam(self.net_.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()

        train_loader  = self._make_loader(X_train, y_train, shuffle=True)
        best_val_loss = float("inf")
        no_improve    = 0

        tag = "BiLSTM" if self.bidirectional else "LSTM"
        print(f"  [{tag}] Training on {N} samples | device={self.device} | epochs={self.epochs}")

        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net_(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_loss = total_loss / N

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

        if hasattr(self, "_best_state"):
            self.net_.load_state_dict(self._best_state)

        self.is_fitted = True
        return self

    def _forward(self, X: np.ndarray) -> "torch.Tensor":
        self.net_.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.net_(tx)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X).argmax(1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return torch.softmax(self._forward(X), dim=1).cpu().numpy()

    def get_params(self) -> dict:
        return {"hidden_size": self.hidden_size, "num_layers": self.num_layers,
                "bidirectional": self.bidirectional, "epochs": self.epochs}
