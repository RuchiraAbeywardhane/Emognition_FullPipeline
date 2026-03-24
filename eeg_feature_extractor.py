"""
EEG Feature Extractor — Emognition Pipeline
============================================================
Extracts a flat feature vector from each EEG window.

Input  : X_raw  (N, W, C)  — N windows, W samples, C channels
Output : X_feat (N, F)     — F = C × 26 features per window

Features per channel (26 total)
--------------------------------
  Spectral  (10) : PSD + Differential Entropy for each of 5 bands
                   delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz),
                   beta (13–30 Hz), gamma (30–45 Hz)
  Temporal  (16) : mean, std, var, rms, skew, kurtosis, peak-to-peak,
                   zero-crossing-rate, energy, Hjorth mobility,
                   Hjorth complexity, IQR, max, min, median, SNR

Usage
-----
    from eeg_feature_extractor import extract_eeg_features, get_feature_names
    from config import CONFIG

    # X_raw : (N, W, 4) from data_loader.load_eeg_data()
    X_feat = extract_eeg_features(X_raw, CONFIG)   # → (N, 104)
    names  = get_feature_names(CONFIG)              # → list of 104 strings

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Channel names (must match data_loader.EEG_COLS order)
# ---------------------------------------------------------------------------
CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

# ---------------------------------------------------------------------------
# Default frequency bands
# ---------------------------------------------------------------------------
BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _cfg_get(config, key: str, default):
    """Read from config object or dict."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _bandpower(x: np.ndarray, fs: float, lo: float, hi: float, eps: float) -> float:
    """
    Estimate average power in a frequency band using FFT.

    Parameters
    ----------
    x   : 1-D signal array (W samples).
    fs  : Sampling frequency in Hz.
    lo  : Lower band edge in Hz.
    hi  : Upper band edge in Hz.
    eps : Floor value to avoid log(0).

    Returns
    -------
    Power (float), always >= eps.
    """
    N    = len(x)
    fft  = np.fft.rfft(x, n=N)
    freq = np.fft.rfftfreq(N, d=1.0 / fs)
    idx  = np.where((freq >= lo) & (freq < hi))[0]
    if len(idx) == 0:
        return eps
    return float(np.mean(np.abs(fft[idx]) ** 2)) + eps


def _spectral_features(x: np.ndarray, fs: float, eps: float) -> List[float]:
    """
    Compute PSD and Differential Entropy for each frequency band.

    Returns 10 values: [psd_delta, psd_theta, …, de_delta, de_theta, …]
    """
    psd_vals = []
    de_vals  = []
    for lo, hi in BANDS.values():
        bp = _bandpower(x, fs, lo, hi, eps)
        psd_vals.append(bp)
        de_vals.append(0.5 * np.log(2 * np.pi * np.e * bp))
    return psd_vals + de_vals   # 5 + 5 = 10


def _temporal_features(x: np.ndarray, W: int, eps: float) -> List[float]:
    """
    Compute 16 time-domain features from a single-channel window.

    Returns 16 values in the order:
        mean, std, var, rms, skew, kurtosis, p2p, zcr,
        energy, mobility, complexity, iqr, max, min, median, snr
    """
    mean_v   = float(np.mean(x))
    std_v    = float(np.std(x))
    var_v    = float(np.var(x))
    rms_v    = float(np.sqrt(np.mean(x ** 2)))
    skew_v   = float(skew(x))
    kurt_v   = float(kurtosis(x))
    p2p_v    = float(np.ptp(x))
    zcr_v    = float(np.sum((x[:-1] * x[1:]) < 0)) / W

    energy   = float(np.sum(x ** 2))

    # Hjorth parameters
    diff1    = np.diff(x)
    diff2    = np.diff(diff1)
    mob_v    = float(np.std(diff1) / (std_v + eps))
    comp_v   = float(np.std(diff2) / (np.std(diff1) + eps))

    q25, q75 = np.percentile(x, [25, 75])
    iqr_v    = float(q75 - q25)
    max_v    = float(np.max(x))
    min_v    = float(np.min(x))
    median_v = float(np.median(x))
    snr_v    = float(mean_v / (std_v + eps))

    return [
        mean_v, std_v, var_v, rms_v, skew_v, kurt_v,
        p2p_v, zcr_v, energy, mob_v, comp_v, iqr_v,
        max_v, min_v, median_v, snr_v,
    ]


def _extract_window(window: np.ndarray, fs: float, eps: float) -> List[float]:
    """
    Extract all features from a single window of shape (W, C).

    Returns C × 26 floats, channels interleaved:
        [ch0_feat0…feat25, ch1_feat0…feat25, …]
    """
    W, C    = window.shape
    all_feats = []
    for c in range(C):
        x = window[:, c].astype(np.float64)
        # Replace any remaining NaN/Inf with 0
        x = np.where(np.isfinite(x), x, 0.0)
        spectral = _spectral_features(x, fs, eps)   # 10
        temporal = _temporal_features(x, W, eps)    # 16
        all_feats.extend(spectral)
        all_feats.extend(temporal)
    return all_feats   # C × 26


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
    Extract a flat feature matrix from raw EEG windows.

    Parameters
    ----------
    X_raw  : (N, W, C)  Raw EEG windows  float32/float64.
    config : Config object or dict.  Reads:
                 EEG_FS  (float)  sampling rate  [default: 256.0]
                 EPS     (float)  stability eps  [default: 1e-12]
    fs     : Fallback sampling rate if not in config.
    eps    : Fallback eps if not in config.

    Returns
    -------
    X_feat : (N, C × 26)  float32 feature matrix.
    """
    fs  = float(_cfg_get(config, "EEG_FS", fs))
    eps = float(_cfg_get(config, "EPS",    eps))

    if X_raw.ndim != 3:
        raise ValueError(
            f"X_raw must be shape (N, W, C), got {X_raw.shape}"
        )

    N, W, C = X_raw.shape
    n_feats = C * 26

    X_feat = np.zeros((N, n_feats), dtype=np.float32)

    for i in range(N):
        X_feat[i] = _extract_window(X_raw[i], fs, eps)

    return X_feat   # (N, C*26)


def get_feature_names(config=None, n_channels: int = 4) -> List[str]:
    """
    Return the ordered list of feature names matching extract_eeg_features output.

    Parameters
    ----------
    config     : Config object or dict (reads channel count if available).
    n_channels : Number of EEG channels  [default: 4].

    Returns
    -------
    List of strings, length = n_channels × 26.
    Example: ['TP9_psd_delta', 'TP9_psd_theta', …, 'TP10_snr']
    """
    spectral_names = (
        [f"psd_{b}" for b in BANDS]  +   # 5
        [f"de_{b}"  for b in BANDS]       # 5
    )
    temporal_names = [
        "mean", "std", "var", "rms", "skew", "kurtosis",
        "p2p", "zcr", "energy", "mobility", "complexity",
        "iqr", "max", "min", "median", "snr",
    ]
    per_channel = spectral_names + temporal_names   # 26

    ch_names = CHANNEL_NAMES[:n_channels]
    return [f"{ch}_{feat}" for ch in ch_names for feat in per_channel]
