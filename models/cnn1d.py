"""
cnn1d.py — 1D Convolutional Neural Network Classifier
============================================================
Operates on raw EEG windows: (N, W, C).
Learns spatial-temporal patterns directly from the signal.

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

class _CNN1DNet(nn.Module):
    """
    Stacked 1-D convolutions → global avg pool → classifier head.
    Input  : (batch, C, W)   channels-first
    Output : (batch, num_classes)
    """

    def __init__(self, n_channels: int, win_len: int,
                 filters: List[int], kernel_size: int,
                 pool_size: int, dropout: float, num_classes: int):
        super().__init__()

        conv_blocks = []
        in_ch = n_channels
        for out_ch in filters:
            conv_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv  = nn.Sequential(*conv_blocks)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.head  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.conv(x)            # (B, C_last, W')
        x = self.pool(x)            # (B, C_last, 1)
        return self.head(x)         # (B, num_classes)


# ===========================================================================
# BaseModel wrapper
# ===========================================================================

class CNN1DModel(BaseModel):
    """
    1D CNN classifier.  Input: raw windows (N, W, C).

    Config keys  (CFG["models"]["architectures"]["cnn1d"])
    -------------------------------------------------------
    filters      : list of int   out-channels per conv layer  [[32,64,128]]
    kernel_size  : int           conv kernel length           [3]
    pool_size    : int           max-pool stride              [2]
    dropout      : float                                      [0.3]

    Training config  (CFG["training"])
    ------------------------------------
    epochs / batch_size / learning_rate / weight_decay / patience
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CNN1DModel.")

        arch     = {}
        training = {}
        if isinstance(config, dict):
            arch     = config.get("models", {}).get("architectures", {}).get("cnn1d", {})
            training = config.get("training", {})

        self.filters     = arch.get("filters",     [32, 64, 128])
        self.kernel_size = arch.get("kernel_size", 3)
        self.pool_size   = arch.get("pool_size",   2)
        self.dropout     = arch.get("dropout",     0.3)

        self.epochs       = training.get("epochs",        50)
        self.batch_size   = training.get("batch_size",    32)
        self.lr           = training.get("learning_rate", 1e-3)
        self.weight_decay = training.get("weight_decay",  1e-4)
        self.patience     = training.get("patience",      10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_: Optional[_CNN1DNet] = None

    def _build(self, n_channels: int, win_len: int) -> None:
        self.net_ = _CNN1DNet(
            n_channels, win_len, self.filters,
            self.kernel_size, self.pool_size,
            self.dropout, self.num_classes,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        # X: (N, W, C) → transpose to (N, C, W) for Conv1d
        tx = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(tx, ty),
                          batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "CNN1DModel":

        N, W, C = X_train.shape
        self._build(C, W)
        optimizer = torch.optim.Adam(self.net_.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        best_val_loss = float("inf")
        no_improve    = 0

        print(f"  [CNN1D] Training on {N} samples | device={self.device} | epochs={self.epochs}")

        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net_(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_loss = total_loss / N

            if X_val is not None and y_val is not None:
                self.net_.eval()
                with torch.no_grad():
                    xv = torch.tensor(X_val.transpose(0, 2, 1),
                                      dtype=torch.float32).to(self.device)
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
            tx = torch.tensor(X.transpose(0, 2, 1),
                              dtype=torch.float32).to(self.device)
            return self.net_(tx)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X).argmax(1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return torch.softmax(self._forward(X), dim=1).cpu().numpy()

    def get_params(self) -> dict:
        return {"filters": self.filters, "kernel_size": self.kernel_size,
                "dropout": self.dropout, "epochs": self.epochs}
