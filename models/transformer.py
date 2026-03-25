"""
transformer.py — Transformer Encoder Classifier
============================================================
Operates on raw EEG windows: (N, W, C).
Multi-head self-attention captures long-range temporal dependencies.

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import math
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


def _compute_class_weights(y: np.ndarray, num_classes: int) -> "torch.Tensor":
    """Compute inverse-frequency class weights and return as a CPU tensor."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ===========================================================================
# PyTorch Module
# ===========================================================================

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to the input embeddings."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.dropout(x + self.pe[:, : x.size(1)])


class _TransformerNet(nn.Module):
    """
    Linear projection → positional encoding → Transformer encoder
    → mean pooling → classifier head.

    Input  : (batch, W, C)   time-first
    Output : (batch, num_classes)
    """

    def __init__(self, n_channels: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int,
                 dropout: float, num_classes: int):
        super().__init__()

        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.input_proj(x)      # (B, W, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)         # (B, W, d_model)
        x = x.mean(dim=1)           # mean pooling over time
        return self.head(x)


# ===========================================================================
# BaseModel wrapper
# ===========================================================================

class TransformerModel(BaseModel):
    """
    Transformer encoder classifier.  Input: raw windows (N, W, C).

    Config keys  (CFG["models"]["architectures"]["transformer"])
    -------------------------------------------------------------
    d_model         : int   embedding dimension           [64]
    nhead           : int   attention heads               [4]
    num_layers      : int   encoder layers                [2]
    dim_feedforward : int   FFN inner dimension           [256]
    dropout         : float                               [0.1]

    Training config  (CFG["training"])
    ------------------------------------
    epochs / batch_size / learning_rate / weight_decay / patience
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TransformerModel.")

        arch     = {}
        training = {}
        if isinstance(config, dict):
            arch     = config.get("models", {}).get("architectures", {}).get("transformer", {})
            training = config.get("training", {})

        self.d_model         = arch.get("d_model",         64)
        self.nhead           = arch.get("nhead",           4)
        self.num_layers      = arch.get("num_layers",      2)
        self.dim_feedforward = arch.get("dim_feedforward", 256)
        self.dropout         = arch.get("dropout",         0.1)

        self.epochs       = training.get("epochs",        50)
        self.batch_size   = training.get("batch_size",    32)
        self.lr           = training.get("learning_rate", 1e-3)
        self.weight_decay = training.get("weight_decay",  1e-4)
        self.patience     = training.get("patience",      10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_: Optional[_TransformerNet] = None

    def _build(self, n_channels: int) -> None:
        self.net_ = _TransformerNet(
            n_channels, self.d_model, self.nhead,
            self.num_layers, self.dim_feedforward,
            self.dropout, self.num_classes,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        tx = torch.tensor(X, dtype=torch.float32)   # (N, W, C) — time-first OK
        ty = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(tx, ty),
                          batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "TransformerModel":

        from collections import Counter
        dist = Counter(y_train.tolist())

        N, W, C = X_train.shape
        self._build(C)
        optimizer = torch.optim.AdamW(self.net_.parameters(),
                                      lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6)

        class_weights = _compute_class_weights(y_train, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_loader  = self._make_loader(X_train, y_train, shuffle=True)
        best_val_loss = float("inf")
        no_improve    = 0

        print(f"  [Transformer] Training on {N} samples | "
              f"device={self.device} | epochs={self.epochs}")
        print(f"  [Transformer] Class distribution : {dict(dist)}")
        print(f"  [Transformer] Class weights      : {class_weights.tolist()}")

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
            scheduler.step()

            if X_val is not None and y_val is not None:
                self.net_.eval()
                with torch.no_grad():
                    xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    yv = torch.tensor(y_val, dtype=torch.long).to(self.device)
                    val_loss = criterion(self.net_(xv), yv).item()
                    val_acc  = (self.net_(xv).argmax(1) == yv).float().mean().item()

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
        return {"d_model": self.d_model, "nhead": self.nhead,
                "num_layers": self.num_layers, "dropout": self.dropout,
                "epochs": self.epochs}
