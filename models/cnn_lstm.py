"""
cnn_lstm.py — CNN + Bidirectional LSTM Classifier
============================================================
Operates on raw EEG windows: (N, W, C).
CNN extracts local features, BiLSTM captures temporal dynamics.

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

class _CNNLSTMNet(nn.Module):
    """
    CNN1D feature extractor → BiLSTM sequence model → classifier head.
    Input  : (batch, C, W)   channels-first
    Output : (batch, num_classes)
    """

    def __init__(self, n_channels: int, cnn_filters: List[int],
                 cnn_kernel: int, lstm_hidden: int,
                 dropout: float, num_classes: int):
        super().__init__()

        cnn_blocks = []
        in_ch = n_channels
        for out_ch in cnn_filters:
            cnn_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel,
                          padding=cnn_kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_blocks)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.cnn(x)                              # (B, C_last, W')
        x = x.permute(0, 2, 1)                       # (B, W', C_last)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)        # (B, hidden*2)
        return self.head(self.dropout(h))


# ===========================================================================
# BaseModel wrapper
# ===========================================================================

class CNNLSTMModel(BaseModel):
    """
    CNN + BiLSTM classifier.  Input: raw windows (N, W, C).

    Config keys  (CFG["models"]["architectures"]["cnn_lstm"])
    ----------------------------------------------------------
    cnn_filters     : list of int   conv layer out-channels  [[32, 64]]
    cnn_kernel_size : int           conv kernel length       [3]
    lstm_hidden     : int           LSTM hidden units        [128]
    dropout         : float                                  [0.3]

    Training config  (CFG["training"])
    ------------------------------------
    epochs / batch_size / learning_rate / weight_decay / patience
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__(num_classes, config)
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CNNLSTMModel.")

        arch     = {}
        training = {}
        if isinstance(config, dict):
            arch     = config.get("models", {}).get("architectures", {}).get("cnn_lstm", {})
            training = config.get("training", {})

        self.cnn_filters  = arch.get("cnn_filters",      [32, 64])
        self.cnn_kernel   = arch.get("cnn_kernel_size",  3)
        self.lstm_hidden  = arch.get("lstm_hidden",      128)
        self.dropout      = arch.get("dropout",          0.3)

        self.epochs       = training.get("epochs",        50)
        self.batch_size   = training.get("batch_size",    32)
        self.lr           = training.get("learning_rate", 1e-3)
        self.weight_decay = training.get("weight_decay",  1e-4)
        self.patience     = training.get("patience",      10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_: Optional[_CNNLSTMNet] = None

    def _build(self, n_channels: int) -> None:
        self.net_ = _CNNLSTMNet(
            n_channels, self.cnn_filters, self.cnn_kernel,
            self.lstm_hidden, self.dropout, self.num_classes,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        # (N, W, C) → (N, C, W) for CNN
        tx = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(tx, ty),
                          batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "CNNLSTMModel":

        from collections import Counter
        dist = Counter(y_train.tolist())

        N, W, C = X_train.shape
        self._build(C)
        optimizer = torch.optim.Adam(self.net_.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6)

        class_weights = _compute_class_weights(y_train, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_loader  = self._make_loader(X_train, y_train, shuffle=True)
        best_val_loss = float("inf")
        no_improve    = 0

        print(f"  [CNN-LSTM] Training on {N} samples | device={self.device} | epochs={self.epochs}")
        print(f"  [CNN-LSTM] Class distribution : {dict(dist)}")
        print(f"  [CNN-LSTM] Class weights      : {class_weights.tolist()}")

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
        return {"cnn_filters": self.cnn_filters, "lstm_hidden": self.lstm_hidden,
                "dropout": self.dropout, "epochs": self.epochs}
