"""
EEG Feature Extractor — Emognition Pipeline
============================================================
Extracts features from raw EEG windows.

Three modes (set via config.FEATURE_MODE)
------------------------------------------
  'statistical' : Hand-crafted features only  → (N, C×39 + 30 + 10)
  'deep'        : Learned CNN/LSTM/Transformer embedding → (N, embed_dim)
  'combined'    : Both concatenated

Statistical features
--------------------
  Per channel (39):
    Spectral   (10) : PSD + Differential Entropy × 5 bands
    Temporal   (16) : mean, std, var, rms, skew, kurtosis, p2p, zcr,
                      energy, Hjorth mobility, Hjorth complexity,
                      IQR, max, min, median, SNR
    Wavelet     (5) : db4 subband energies (5 levels)
    Entropy     (2) : spectral entropy, permutation entropy
    Hjorth      (3) : activity, mobility, complexity  (full triple)
    Ratios      (3) : alpha/beta, alpha/theta, theta/beta  (per channel)

  Cross-channel (30):
    Coherence per band × 6 channel pairs  (5 × 6 = 30)

  Asymmetry (10):
    AF7–AF8 band power diff × 5 bands
    TP9–TP10 band power diff × 5 bands

  Total: 4×39 + 30 + 10 = 196

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch, coherence as sp_coherence
from typing import List, Optional

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

BAND_ITEMS = list(BANDS.items())   # ordered list of (name, (lo, hi))

# 6 unique channel pairs for coherence
CH_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

N_PER_CHANNEL      = 39   # spectral(10) + temporal(16) + wavelet(5) + entropy(2) + hjorth(3) + ratios(3)
N_COHERENCE        = 30   # 5 bands × 6 pairs
N_ASYMMETRY        = 10   # 5 bands × 2 pairs (AF7-AF8, TP9-TP10)
N_STAT_FEATURES    = 4 * N_PER_CHANNEL + N_COHERENCE + N_ASYMMETRY  # 196


# ===========================================================================
# CONFIG HELPER
# ===========================================================================

def _cfg(config, key: str, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ===========================================================================
# STATISTICAL FEATURE HELPERS
# ===========================================================================

def _bandpower_fft(x: np.ndarray, fs: float, lo: float, hi: float, eps: float) -> float:
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
    for _, (lo, hi) in BAND_ITEMS:
        bp = _bandpower_fft(x, fs, lo, hi, eps)
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
        float(np.sqrt(np.mean(x ** 2))),           # RMS
        float(skew(x)),
        float(kurtosis(x)),
        float(np.ptp(x)),                           # peak-to-peak
        float(np.sum((x[:-1] * x[1:]) < 0)) / W,  # ZCR
        float(np.sum(x ** 2)),                      # energy
        float(np.std(diff1) / (std_v + eps)),       # Hjorth mobility
        float(np.std(diff2) / (np.std(diff1) + eps)),  # Hjorth complexity
        float(q75 - q25),                           # IQR
        float(np.max(x)),
        float(np.min(x)),
        float(np.median(x)),
        float(mean_v / (std_v + eps)),              # SNR proxy
    ]


def _hjorth_full(x: np.ndarray, eps: float) -> List[float]:
    """3 Hjorth parameters: activity, mobility, complexity."""
    d1  = np.diff(x)
    d2  = np.diff(d1)
    act = float(np.var(x))
    mob = float(np.sqrt(np.var(d1) / (act + eps)))
    var_d1 = float(np.var(d1))
    comp = float(np.sqrt(np.var(d2) / (var_d1 + eps)) / (mob + eps))
    return [act, mob, comp]


def _spectral_entropy(x: np.ndarray, fs: float, nperseg: int = 256) -> float:
    """Normalised spectral entropy using Welch PSD."""
    _, psd = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    psd_n  = psd / (psd.sum() + 1e-12)
    psd_n  = psd_n[psd_n > 0]
    return float(-np.sum(psd_n * np.log2(psd_n))) if len(psd_n) > 0 else 0.0


def _permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Permutation entropy of order `order`."""
    n = len(x)
    if n < (order - 1) * delay + 1:
        return 0.0
    idx  = np.arange(n - (order - 1) * delay)
    cols = np.column_stack([x[idx + d * delay] for d in range(order)])
    perms = np.argsort(cols, axis=1)
    encoded = np.zeros(perms.shape[0], dtype=np.int64)
    for i in range(order):
        encoded = encoded * order + perms[:, i]
    _, counts = np.unique(encoded, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _wavelet_energies(x: np.ndarray, wavelet: str = "db4", level: int = 5) -> List[float]:
    """5 subband energies from a DWT decomposition."""
    try:
        import pywt
        max_lev = pywt.dwt_max_level(len(x), wavelet)
        coeffs  = pywt.wavedec(x, wavelet, level=min(level, max_lev))
        energies = [float(np.mean(c ** 2)) for c in coeffs[:5]]
        while len(energies) < 5:
            energies.append(0.0)
        return energies
    except Exception:
        return [0.0] * 5


def _band_ratio_features(x: np.ndarray, fs: float, eps: float) -> List[float]:
    """3 asymmetry/ratio features: alpha/beta, alpha/theta, theta/beta."""
    bp = {}
    for name, (lo, hi) in BAND_ITEMS:
        bp[name] = _bandpower_fft(x, fs, lo, hi, eps)
    a, b, t = bp["alpha"], bp["beta"], bp["theta"]
    return [
        a / (b + eps),
        a / (t + eps),
        t / (b + eps),
    ]


def _per_channel_features(x: np.ndarray, fs: float, eps: float) -> List[float]:
    """
    All 39 per-channel features:
      spectral(10) + temporal(16) + hjorth_full(3) + entropy(2)
      + wavelet(5) + ratios(3)
    """
    feats = []
    feats.extend(_spectral_features(x, fs, eps))          # 10
    feats.extend(_temporal_features(x, len(x), eps))      # 16
    feats.extend(_hjorth_full(x, eps))                     # 3
    feats.append(_spectral_entropy(x, fs))                 # 1
    feats.append(_permutation_entropy(x))                  # 1
    feats.extend(_wavelet_energies(x))                     # 5
    feats.extend(_band_ratio_features(x, fs, eps))         # 3
    return feats                                            # = 39


def _coherence_features(channels: List[np.ndarray], fs: float) -> List[float]:
    """30 coherence features: mean coherence per band × 6 channel pairs."""
    feats = []
    nperseg = min(256, len(channels[0]))
    for ci, cj in CH_PAIRS:
        try:
            f_coh, coh = sp_coherence(channels[ci], channels[cj],
                                      fs=fs, nperseg=nperseg)
            for _, (lo, hi) in BAND_ITEMS:
                mask = (f_coh >= lo) & (f_coh < hi)
                feats.append(float(np.mean(coh[mask])) if mask.any() else 0.0)
        except Exception:
            feats.extend([0.0] * 5)
    return feats  # 30


def _asymmetry_features(channels: List[np.ndarray], fs: float, eps: float) -> List[float]:
    """
    10 asymmetry features:
      AF7(idx1) – AF8(idx2) per band × 5
      TP9(idx0) – TP10(idx3) per band × 5
    """
    feats = []
    # AF7 vs AF8  (indices 1, 2)
    for _, (lo, hi) in BAND_ITEMS:
        p1 = _bandpower_fft(channels[1], fs, lo, hi, eps)
        p2 = _bandpower_fft(channels[2], fs, lo, hi, eps)
        feats.append(p1 - p2)
    # TP9 vs TP10 (indices 0, 3)
    for _, (lo, hi) in BAND_ITEMS:
        p0 = _bandpower_fft(channels[0], fs, lo, hi, eps)
        p3 = _bandpower_fft(channels[3], fs, lo, hi, eps)
        feats.append(p0 - p3)
    return feats  # 10


def _safe(v: float) -> float:
    """Replace non-finite values with 0."""
    return 0.0 if not np.isfinite(v) else v


def _statistical_features_window(window: np.ndarray, fs: float, eps: float) -> np.ndarray:
    """
    Extract all 196 statistical features from one (W, C) window.

    Layout
    ------
    [ch0_39 | ch1_39 | ch2_39 | ch3_39 | coherence_30 | asymmetry_10]
    """
    W, C = window.shape
    channels = [
        np.where(np.isfinite(window[:, c]), window[:, c], 0.0).astype(np.float64)
        for c in range(C)
    ]

    feats: List[float] = []

    # Per-channel block
    for ch in channels:
        feats.extend(_per_channel_features(ch, fs, eps))  # 39 each

    # Cross-channel coherence
    feats.extend(_coherence_features(channels, fs))        # 30

    # Asymmetry
    if C >= 4:
        feats.extend(_asymmetry_features(channels, fs, eps))  # 10
    else:
        feats.extend([0.0] * N_ASYMMETRY)

    arr = np.array([_safe(v) for v in feats], dtype=np.float32)
    return arr


def _extract_statistical(X_raw: np.ndarray, fs: float, eps: float) -> np.ndarray:
    """
    Extract statistical features from all windows.

    Parameters
    ----------
    X_raw : (N, W, C)

    Returns
    -------
    (N, N_STAT_FEATURES)  float32   — N_STAT_FEATURES = 196
    """
    N, W, C = X_raw.shape
    n_feats  = C * N_PER_CHANNEL + N_COHERENCE + N_ASYMMETRY
    out      = np.zeros((N, n_feats), dtype=np.float32)
    for i in range(N):
        out[i] = _statistical_features_window(X_raw[i], fs, eps)
    return out


# ===========================================================================
# DEEP LEARNING FEATURE EXTRACTORS  (PyTorch)  — unchanged
# ===========================================================================

class _CNN1DExtractor(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32,  kernel_size=7, padding=3), nn.BatchNorm1d(32),  nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32,         64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64,         128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class _LSTMExtractor(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.proj = nn.Linear(hidden * 2, embed_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h)


class _CNNLSTMExtractor(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32,         64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.proj = nn.Linear(hidden * 2, embed_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h)


class _TransformerExtractor(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj    = nn.Linear(d_model, embed_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.proj(x)


_DEEP_MODELS = {
    "cnn1d":       _CNN1DExtractor,
    "lstm":        _LSTMExtractor,
    "cnn_lstm":    _CNNLSTMExtractor,
    "transformer": _TransformerExtractor,
}
_model_cache: dict = {}


def _get_deep_model(name: str, n_channels: int, embed_dim: int):
    key = (name, n_channels, embed_dim)
    if key not in _model_cache:
        if name not in _DEEP_MODELS:
            raise ValueError(f"Unknown deep extractor '{name}'. Choose from: {list(_DEEP_MODELS.keys())}")
        _model_cache[key] = _DEEP_MODELS[name](n_channels, embed_dim).eval()
    return _model_cache[key]


def _extract_deep(X_raw, extractor, embed_dim, batch_size, device):
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for deep feature extraction.")
    N, W, C = X_raw.shape
    dev   = torch.device(device)
    model = _get_deep_model(extractor, C, embed_dim).to(dev)
    channels_first = extractor in ("cnn1d", "cnn_lstm")
    embeddings = np.zeros((N, embed_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            t     = torch.from_numpy(X_raw[start:end].astype(np.float32)).to(dev)
            if channels_first:
                t = t.permute(0, 2, 1)
            embeddings[start:end] = model(t).cpu().numpy()
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

    Modes (config.FEATURE_MODE)
    ---------------------------
    'statistical'  → (N, 196)          hand-crafted only
    'deep'         → (N, DEEP_EMBED_DIM)
    'combined'     → (N, 196 + DEEP_EMBED_DIM)
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
        stat = _extract_statistical(X_raw, fs, eps)
        deep = _extract_deep(X_raw, extractor, embed_dim, batch_size, device)
        return np.concatenate([stat, deep], axis=1)
    else:
        raise ValueError(f"Unknown FEATURE_MODE '{mode}'. Choose 'statistical', 'deep', or 'combined'.")


def get_feature_names(config=None, n_channels: int = 4) -> List[str]:
    """Return ordered feature names matching extract_eeg_features output."""
    mode      = str(_cfg(config, "FEATURE_MODE",   "statistical")).lower()
    embed_dim = int(_cfg(config, "DEEP_EMBED_DIM",  128))
    extractor = str(_cfg(config, "DEEP_EXTRACTOR",  "cnn1d"))

    band_names     = list(BANDS.keys())
    spectral_names = [f"psd_{b}" for b in band_names] + [f"de_{b}" for b in band_names]
    temporal_names = [
        "mean", "std", "var", "rms", "skew", "kurtosis",
        "p2p", "zcr", "energy", "mobility", "complexity",
        "iqr", "max", "min", "median", "snr",
    ]
    hjorth_names   = ["hjorth_act", "hjorth_mob", "hjorth_comp"]
    entropy_names  = ["spectral_entropy", "perm_entropy"]
    wavelet_names  = [f"wavelet_{i}" for i in range(5)]
    ratio_names    = ["ratio_alpha_beta", "ratio_alpha_theta", "ratio_theta_beta"]

    per_ch = (spectral_names + temporal_names + hjorth_names +
              entropy_names + wavelet_names + ratio_names)   # 39

    ch_names   = CHANNEL_NAMES[:n_channels]
    stat_names = [f"{ch}_{f}" for ch in ch_names for f in per_ch]

    # Coherence: pair_band
    pair_names = [f"ch{ci}ch{cj}" for ci, cj in CH_PAIRS]
    coh_names  = [f"coh_{p}_{b}" for p in pair_names for b in band_names]  # 30

    # Asymmetry
    asym_names = (
        [f"asym_AF7AF8_{b}"  for b in band_names] +   # 5
        [f"asym_TP9TP10_{b}" for b in band_names]      # 5
    )

    stat_all  = stat_names + coh_names + asym_names   # 196
    deep_names = [f"{extractor}_embed_{i}" for i in range(embed_dim)]

    if mode == "statistical":
        return stat_all
    elif mode == "deep":
        return deep_names
    elif mode == "combined":
        return stat_all + deep_names
    else:
        return stat_all


def clear_model_cache() -> None:
    """Release cached deep extractor models from memory."""
    global _model_cache
    _model_cache.clear()
    if _TORCH_AVAILABLE:
        torch.cuda.empty_cache()
    print("[feature_extractor] Model cache cleared.")
