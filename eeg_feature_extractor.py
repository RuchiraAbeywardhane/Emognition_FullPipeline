"""
EEG Feature Extractor — Emognition Pipeline
============================================================
Extracts features from raw EEG windows.

Three modes (set via config.FEATURE_MODE)
------------------------------------------
  'statistical' : Hand-crafted features only  → (N, C×26)
  'deep'        : Learned CNN/LSTM/Transformer embedding → (N, embed_dim)
  'combined'    : Both concatenated → (N, C×26 + embed_dim)

Statistical features per channel (26 total)
--------------------------------------------
  Spectral  (10) : PSD + Differential Entropy × 5 bands
                   delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz),
                   beta (13–30 Hz), gamma (30–45 Hz)
  Temporal  (16) : mean, std, var, rms, skew, kurtosis, p2p, zcr,
                   energy, Hjorth mobility, Hjorth complexity,
                   IQR, max, min, median, SNR

Deep extractors  (all output embedding of size DEEP_EMBED_DIM)
---------------------------------------------------------------
  'cnn1d'       : Stacked 1-D convolutions + global average pooling
  'lstm'        : Bidirectional LSTM, last hidden state
  'cnn_lstm'    : CNN1D → BiLSTM pipeline
  'transformer' : Multi-head self-attention encoder + mean pooling

Usage
-----
    from eeg_feature_extractor import extract_eeg_features, get_feature_names
    from config import CONFIG

    X_feat = extract_eeg_features(X_raw, CONFIG)
    names  = get_feature_names(CONFIG)

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Optional

# ---------------------------------------------------------------------------
# PyTorch — optional (only needed for deep / combined modes)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

N_STAT_FEATURES_PER_CHANNEL = 26   # 10 spectral + 16 temporal


# ===========================================================================
# CONFIG HELPER
# ===========================================================================

def _cfg(config, key: str, default):
    """Read a value from a config object or dict."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ===========================================================================
# STATISTICAL FEATURE HELPERS
# ===========================================================================

def _bandpower(x: np.ndarray, fs: float, lo: float, hi: float, eps: float) -> float:
    """Average FFT power in a frequency band."""
    N    = len(x)
    fft  = np.fft.rfft(x, n=N)
    freq = np.fft.rfftfreq(N, d=1.0 / fs)
    idx  = np.where((freq >= lo) & (freq < hi))[0]
    if len(idx) == 0:
        return eps
    return float(np.mean(np.abs(fft[idx]) ** 2)) + eps


def _spectral_features(x: np.ndarray, fs: float, eps: float) -> List[float]:
    """10 spectral features: [psd×5, de×5]."""
    psd_vals, de_vals = [], []
    for lo, hi in BANDS.values():
        bp = _bandpower(x, fs, lo, hi, eps)
        psd_vals.append(bp)
        de_vals.append(0.5 * np.log(2 * np.pi * np.e * bp))
    return psd_vals + de_vals


def _temporal_features(x: np.ndarray, W: int, eps: float) -> List[float]:
    """16 temporal features."""
    mean_v = float(np.mean(x))
    std_v  = float(np.std(x))
    diff1  = np.diff(x)
    diff2  = np.diff(diff1)
    q25, q75 = np.percentile(x, [25, 75])
    return [
        mean_v,
        std_v,
        float(np.var(x)),
        float(np.sqrt(np.mean(x ** 2))),
        float(skew(x)),
        float(kurtosis(x)),
        float(np.ptp(x)),
        float(np.sum((x[:-1] * x[1:]) < 0)) / W,
        float(np.sum(x ** 2)),
        float(np.std(diff1) / (std_v + eps)),           # Hjorth mobility
        float(np.std(diff2) / (np.std(diff1) + eps)),   # Hjorth complexity
        float(q75 - q25),
        float(np.max(x)),
        float(np.min(x)),
        float(np.median(x)),
        float(mean_v / (std_v + eps)),                  # SNR
    ]


def _statistical_features_window(window: np.ndarray, fs: float, eps: float) -> List[float]:
    """Extract C×26 statistical features from one (W, C) window."""
    W, C = window.shape
    feats = []
    for c in range(C):
        x = window[:, c].astype(np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        feats.extend(_spectral_features(x, fs, eps))
        feats.extend(_temporal_features(x, W, eps))
    return feats


def _extract_statistical(X_raw: np.ndarray, fs: float, eps: float) -> np.ndarray:
    """
    Extract statistical features from all windows.

    Parameters
    ----------
    X_raw : (N, W, C)

    Returns
    -------
    (N, C×26)  float32
    """
    N, W, C = X_raw.shape
    out = np.zeros((N, C * N_STAT_FEATURES_PER_CHANNEL), dtype=np.float32)
    for i in range(N):
        out[i] = _statistical_features_window(X_raw[i], fs, eps)
    return out


# ===========================================================================
# DEEP LEARNING FEATURE EXTRACTORS  (PyTorch)
# ===========================================================================

class _CNN1DExtractor(nn.Module):
    """
    Three-layer 1-D CNN feature extractor.

    Input  : (batch, C, W)  — channels-first
    Output : (batch, embed_dim)
    """
    def __init__(self, n_channels: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32,  kernel_size=7, padding=3), nn.BatchNorm1d(32),  nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32,         64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64,         128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):                   # x: (B, C, W)
        x = self.net(x)                     # (B, 128, W')
        x = self.pool(x).squeeze(-1)        # (B, 128)
        return self.proj(x)                 # (B, embed_dim)


class _LSTMExtractor(nn.Module):
    """
    Bidirectional LSTM feature extractor.

    Input  : (batch, W, C)  — time-first
    Output : (batch, embed_dim)
    """
    def __init__(self, n_channels: int, embed_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden,
            num_layers=layers, batch_first=True,
            bidirectional=True, dropout=0.2,
        )
        self.proj = nn.Linear(hidden * 2, embed_dim)

    def forward(self, x):                   # x: (B, W, C)
        _, (h, _) = self.lstm(x)           # h: (layers*2, B, hidden)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden*2)
        return self.proj(h)                # (B, embed_dim)


class _CNNLSTMExtractor(nn.Module):
    """
    CNN1D → BiLSTM pipeline.

    Input  : (batch, C, W)  — channels-first
    Output : (batch, embed_dim)
    """
    def __init__(self, n_channels: int, embed_dim: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32,         64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.2,
        )
        self.proj = nn.Linear(hidden * 2, embed_dim)

    def forward(self, x):                        # x: (B, C, W)
        x = self.cnn(x)                          # (B, 64, W')
        x = x.permute(0, 2, 1)                  # (B, W', 64)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)   # (B, hidden*2)
        return self.proj(h)                      # (B, embed_dim)


class _TransformerExtractor(nn.Module):
    """
    Transformer encoder feature extractor.

    Input  : (batch, W, C)  — time-first
    Output : (batch, embed_dim)
    """
    def __init__(self, n_channels: int, embed_dim: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj    = nn.Linear(d_model, embed_dim)

    def forward(self, x):               # x: (B, W, C)
        x = self.input_proj(x)          # (B, W, d_model)
        x = self.encoder(x)             # (B, W, d_model)
        x = x.mean(dim=1)              # (B, d_model)  — mean pooling
        return self.proj(x)             # (B, embed_dim)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
_DEEP_MODELS = {
    "cnn1d":       _CNN1DExtractor,
    "lstm":        _LSTMExtractor,
    "cnn_lstm":    _CNNLSTMExtractor,
    "transformer": _TransformerExtractor,
}

# Cache so the model is built only once per session
_model_cache: dict = {}


def _get_deep_model(name: str, n_channels: int, embed_dim: int) -> "nn.Module":
    """Build or retrieve a cached deep extractor model."""
    key = (name, n_channels, embed_dim)
    if key not in _model_cache:
        if name not in _DEEP_MODELS:
            raise ValueError(
                f"Unknown deep extractor '{name}'. "
                f"Choose from: {list(_DEEP_MODELS.keys())}"
            )
        _model_cache[key] = _DEEP_MODELS[name](n_channels, embed_dim).eval()
    return _model_cache[key]


def _extract_deep(
    X_raw:     np.ndarray,
    extractor: str,
    embed_dim: int,
    batch_size: int,
    device:    str,
) -> np.ndarray:
    """
    Run a deep extractor over all windows in batches.

    Parameters
    ----------
    X_raw      : (N, W, C)
    extractor  : Model name — 'cnn1d' | 'lstm' | 'cnn_lstm' | 'transformer'
    embed_dim  : Size of the output embedding vector.
    batch_size : GPU batch size.
    device     : 'cuda' or 'cpu'.

    Returns
    -------
    (N, embed_dim)  float32
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep feature extraction. "
            "Install it with:  pip install torch"
        )

    N, W, C = X_raw.shape
    dev     = torch.device(device)
    model   = _get_deep_model(extractor, C, embed_dim).to(dev)

    # CNN models expect (B, C, W); sequence models expect (B, W, C)
    channels_first = extractor in ("cnn1d", "cnn_lstm")

    embeddings = np.zeros((N, embed_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            batch = X_raw[start:end].astype(np.float32)        # (B, W, C)
            t     = torch.from_numpy(batch).to(dev)

            if channels_first:
                t = t.permute(0, 2, 1)                         # (B, C, W)

            out = model(t)                                     # (B, embed_dim)
            embeddings[start:end] = out.cpu().numpy()

    return embeddings


# ===========================================================================
# PUBLIC API
# ===========================================================================

def extract_eeg_features(
    X_raw:  np.ndarray,
    config,
    fs:     float = 256.0,
    eps:    float = 1e-12,
) -> np.ndarray:
    """
    Extract features from raw EEG windows.

    Behaviour is controlled by config.FEATURE_MODE:

        'statistical' (default)
            Hand-crafted features only.
            Returns (N, C×26).

        'deep'
            Learned embedding from config.DEEP_EXTRACTOR model.
            Returns (N, DEEP_EMBED_DIM).

        'combined'
            Statistical + deep embedding concatenated.
            Returns (N, C×26 + DEEP_EMBED_DIM).

    Parameters
    ----------
    X_raw  : (N, W, C)  float32 — raw EEG windows from data_loader.
    config : Config object or dict.  Reads:
                 EEG_FS          float   sampling rate          [256.0]
                 EPS             float   numerical stability    [1e-12]
                 FEATURE_MODE    str     extraction mode        ['statistical']
                 DEEP_EXTRACTOR  str     deep model name        ['cnn1d']
                 DEEP_EMBED_DIM  int     embedding size         [128]
                 DEEP_BATCH_SIZE int     GPU batch size         [256]
                 DEVICE          str     'cuda' or 'cpu'        ['cpu']
    fs     : Fallback sampling rate if not in config.
    eps    : Fallback eps if not in config.

    Returns
    -------
    X_feat : (N, F)  float32
    """
    fs         = float(_cfg(config, "EEG_FS",          fs))
    eps        = float(_cfg(config, "EPS",             eps))
    mode       = str(  _cfg(config, "FEATURE_MODE",    "statistical")).lower()
    extractor  = str(  _cfg(config, "DEEP_EXTRACTOR",  "cnn1d")).lower()
    embed_dim  = int(  _cfg(config, "DEEP_EMBED_DIM",  128))
    batch_size = int(  _cfg(config, "DEEP_BATCH_SIZE", 256))
    device     = str(  _cfg(config, "DEVICE",          "cpu"))

    if X_raw.ndim != 3:
        raise ValueError(f"X_raw must be (N, W, C), got {X_raw.shape}")

    if mode == "statistical":
        return _extract_statistical(X_raw, fs, eps)

    elif mode == "deep":
        return _extract_deep(X_raw, extractor, embed_dim, batch_size, device)

    elif mode == "combined":
        stat  = _extract_statistical(X_raw, fs, eps)
        deep  = _extract_deep(X_raw, extractor, embed_dim, batch_size, device)
        return np.concatenate([stat, deep], axis=1)

    else:
        raise ValueError(
            f"Unknown FEATURE_MODE '{mode}'. "
            "Choose 'statistical', 'deep', or 'combined'."
        )


def get_feature_names(config=None, n_channels: int = 4) -> List[str]:
    """
    Return ordered feature names matching extract_eeg_features output.

    Parameters
    ----------
    config     : Config object or dict.
    n_channels : Number of EEG channels [default: 4].

    Returns
    -------
    List of strings.
      'statistical' → C×26 names  e.g. 'TP9_psd_delta', 'TP9_mean', …
      'deep'        → embed_dim generic names  e.g. 'deep_0', 'deep_1', …
      'combined'    → statistical names + deep names
    """
    mode      = str(_cfg(config, "FEATURE_MODE",   "statistical")).lower()
    embed_dim = int(_cfg(config, "DEEP_EMBED_DIM",  128))
    extractor = str(_cfg(config, "DEEP_EXTRACTOR",  "cnn1d"))

    spectral_names = [f"psd_{b}" for b in BANDS] + [f"de_{b}" for b in BANDS]
    temporal_names = [
        "mean", "std", "var", "rms", "skew", "kurtosis",
        "p2p", "zcr", "energy", "mobility", "complexity",
        "iqr", "max", "min", "median", "snr",
    ]
    per_channel  = spectral_names + temporal_names          # 26
    ch_names     = CHANNEL_NAMES[:n_channels]
    stat_names   = [f"{ch}_{f}" for ch in ch_names for f in per_channel]
    deep_names   = [f"{extractor}_embed_{i}" for i in range(embed_dim)]

    if mode == "statistical":
        return stat_names
    elif mode == "deep":
        return deep_names
    elif mode == "combined":
        return stat_names + deep_names
    else:
        return stat_names


def clear_model_cache() -> None:
    """Release cached deep extractor models from memory."""
    global _model_cache
    _model_cache.clear()
    if _TORCH_AVAILABLE:
        torch.cuda.empty_cache()
    print("[eeg_feature_extractor] Model cache cleared.")
