"""
EEG Data Loader Module — Emognition Dataset
============================================================
Unified loader supporting both RAW and CLEANED versions of
the Emognition dataset.

Dataset Modes
-------------
  raw     : Original JSON files straight from the MUSE headband.
            Includes HSI / HeadBandOn quality columns and raw
            per-sample timestamps.  Baseline reduction is NOT
            applied here — use baseline_reduction.py separately.

  cleaned : Pre-processed CSV/JSON files that have already had
            artefact removal applied upstream.  Feature extraction
            is delegated to eeg_feature_extractor.py.

Features
--------
  - MUSE headband EEG data loading from JSON files (raw mode)
  - CSV loading for cleaned data
  - Quality filtering (HSI, HeadBandOn) — raw mode only
  - Windowing with configurable overlap
  - 26-feature extraction per channel (DE, PSD, temporal stats …)
  - Subject-independent, clip-independent, and random data splits

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import os
import glob
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ---------------------------------------------------------------------------
# Optional dependency — only needed in 'cleaned' mode
# ---------------------------------------------------------------------------
try:
    from eeg_feature_extractor import extract_eeg_features as _external_extract
    _HAS_EXTRACTOR = True
except ImportError:
    _HAS_EXTRACTOR = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MUSE_EEG_COLS   = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
MUSE_HSI_COLS   = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]
DEFAULT_FS      = 256.0          # Hz — MUSE default sample rate
DEFAULT_WIN_SEC = 4.0            # seconds per window
DEFAULT_OVERLAP = 0.5            # 50 % overlap
LABEL_COL       = "label"
SUBJECT_COL     = "subject_id"
TRIAL_COL       = "trial_id"

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def _to_num(x) -> np.ndarray:
    """Convert various input types to a float64 numpy array."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64)
    return np.array(x, dtype=np.float64)


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate NaN values in a 1-D array.
    Edge NaNs are forward/back filled.
    """
    a = a.copy()
    nans = np.isnan(a)
    if not nans.any():
        return a
    idx = np.arange(len(a))
    a[nans] = np.interp(idx[nans], idx[~nans], a[~nans])
    return a


def check_json_structure(data_root: str, num_samples: int = 3) -> None:
    """
    Print the top-level structure of a few JSON files in *data_root*.
    Useful for debugging / understanding new dataset formats.

    Parameters
    ----------
    data_root   : Root directory that contains the JSON files.
    num_samples : How many files to inspect.
    """
    pattern = os.path.join(data_root, "**", "*.json")
    files   = glob.glob(pattern, recursive=True)[:num_samples]
    if not files:
        print(f"[check_json_structure] No JSON files found under '{data_root}'.")
        return
    for fp in files:
        with open(fp, "r") as fh:
            data = json.load(fh)
        print(f"\n--- {fp} ---")
        if isinstance(data, dict):
            for k, v in data.items():
                vtype = type(v).__name__
                vlen  = len(v) if hasattr(v, "__len__") else "scalar"
                print(f"  key='{k}'  type={vtype}  len={vlen}")
        elif isinstance(data, list):
            print(f"  list of {len(data)} items; first item keys: "
                  f"{list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
        else:
            print(f"  type={type(data)}")


# ===========================================================================
# QUALITY FILTERING  (raw mode only)
# ===========================================================================

def _apply_quality_filter(df: pd.DataFrame, hsi_threshold: int = 2) -> pd.DataFrame:
    """
    Remove samples where the MUSE headband contact quality is poor.

    Parameters
    ----------
    df            : Raw DataFrame with HSI_* and HeadBandOn columns.
    hsi_threshold : Samples with any HSI value > threshold are dropped.
                    MUSE HSI: 1 = good, 2 = ok, 4 = bad.
    """
    mask = pd.Series(True, index=df.index)

    if "HeadBandOn" in df.columns:
        mask &= df["HeadBandOn"].astype(float) == 1.0

    present_hsi = [c for c in MUSE_HSI_COLS if c in df.columns]
    for col in present_hsi:
        mask &= df[col].astype(float) <= hsi_threshold

    dropped = (~mask).sum()
    if dropped > 0:
        print(f"  [quality filter] dropped {dropped}/{len(df)} samples.")
    return df[mask].reset_index(drop=True)


# ===========================================================================
# WINDOWING
# ===========================================================================

def _make_windows(
    signal: np.ndarray,
    fs: float,
    win_sec: float,
    overlap: float,
) -> np.ndarray:
    """
    Slice a (T, C) signal into overlapping windows.

    Parameters
    ----------
    signal  : Array of shape (T, C).
    fs      : Sampling frequency in Hz.
    win_sec : Window length in seconds.
    overlap : Fractional overlap [0, 1).

    Returns
    -------
    windows : Array of shape (N, win_samples, C).
    """
    win_samples  = int(win_sec * fs)
    step_samples = int(win_samples * (1.0 - overlap))
    if step_samples < 1:
        step_samples = 1

    T      = signal.shape[0]
    starts = range(0, T - win_samples + 1, step_samples)

    if not list(starts):
        return np.empty((0, win_samples, signal.shape[1]), dtype=signal.dtype)

    windows = np.stack([signal[s : s + win_samples] for s in starts], axis=0)
    return windows  # (N, W, C)


# ===========================================================================
# FEATURE EXTRACTION  (raw mode — 26 features per channel)
# ===========================================================================

def _bandpower(x: np.ndarray, fs: float, lo: float, hi: float, eps: float) -> float:
    """Estimate band power via FFT."""
    N    = len(x)
    fft  = np.fft.rfft(x, n=N)
    freq = np.fft.rfftfreq(N, d=1.0 / fs)
    idx  = np.where((freq >= lo) & (freq < hi))[0]
    if len(idx) == 0:
        return eps
    power = (np.abs(fft[idx]) ** 2).mean()
    return float(power) + eps


def extract_eeg_features(
    X_raw: np.ndarray,
    config: dict,
    fs: float = DEFAULT_FS,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Extract 26 features per channel from EEG windows.

    Parameters
    ----------
    X_raw  : Array of shape (N, W, C) — N windows, W samples, C channels.
    config : Pipeline config dict (uses 'fs' key if present).
    fs     : Sampling rate (overridden by config['fs'] if set).
    eps    : Small constant for numerical stability.

    Returns
    -------
    X_feat : Array of shape (N, C * 26).
    """
    if _HAS_EXTRACTOR:
        return _external_extract(X_raw, config, fs=fs, eps=eps)

    fs = float(config.get("fs", fs))

    bands = {
        "delta": (0.5,  4.0),
        "theta": (4.0,  8.0),
        "alpha": (8.0,  13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    N, W, C = X_raw.shape
    features = []

    for n in range(N):
        win_feats = []
        for c in range(C):
            x = _interp_nan(_to_num(X_raw[n, :, c]))

            # --- Spectral features (5 bands × 2 = 10) ---
            psd, de = [], []
            for lo, hi in bands.values():
                bp = _bandpower(x, fs, lo, hi, eps)
                psd.append(bp)
                de.append(0.5 * np.log(2 * np.pi * np.e * bp))

            # --- Temporal features (16) ---
            mean_v   = float(np.mean(x))
            std_v    = float(np.std(x))
            var_v    = float(np.var(x))
            rms_v    = float(np.sqrt(np.mean(x ** 2)))
            skew_v   = float(skew(x))
            kurt_v   = float(kurtosis(x))
            p2p_v    = float(np.ptp(x))
            zcr_v    = float(((x[:-1] * x[1:]) < 0).sum()) / W
            energy   = float(np.sum(x ** 2))
            mob_v    = float(np.std(np.diff(x)) / (std_v + eps))
            comp_v   = float(
                np.std(np.diff(np.diff(x))) / (np.std(np.diff(x)) + eps)
            )
            q25, q75 = np.percentile(x, [25, 75])
            iqr_v    = float(q75 - q25)
            max_v    = float(np.max(x))
            min_v    = float(np.min(x))
            median_v = float(np.median(x))
            snr_v    = float(mean_v / (std_v + eps))

            temporal = [
                mean_v, std_v, var_v, rms_v, skew_v, kurt_v,
                p2p_v, zcr_v, energy, mob_v, comp_v, iqr_v,
                max_v, min_v, median_v, snr_v,
            ]

            win_feats.extend(psd)       # 5
            win_feats.extend(de)        # 5
            win_feats.extend(temporal)  # 16
            # total per channel = 26

        features.append(win_feats)

    return np.array(features, dtype=np.float32)  # (N, C*26)


# ===========================================================================
# JSON LOADING HELPERS  (raw mode)
# ===========================================================================

def _load_single_json(filepath: str, config: dict) -> Optional[pd.DataFrame]:
    """
    Load a single MUSE JSON recording into a DataFrame.

    Expected JSON formats
    ---------------------
    1. A dict with a 'data' key whose value is a list of sample dicts.
    2. A plain list of sample dicts.
    3. A dict whose values are lists of equal length (column-oriented).
    """
    with open(filepath, "r") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict) and "data" in raw:
        df = pd.DataFrame(raw["data"])
    elif isinstance(raw, list):
        df = pd.DataFrame(raw)
    elif isinstance(raw, dict):
        df = pd.DataFrame(raw)
    else:
        print(f"  [warning] Unrecognised JSON format in '{filepath}' — skipping.")
        return None

    keep     = MUSE_EEG_COLS + MUSE_HSI_COLS
    optional = ["HeadBandOn", "TimeStamp", LABEL_COL, SUBJECT_COL, TRIAL_COL]
    keep    += [c for c in optional if c in df.columns]
    df       = df[[c for c in keep if c in df.columns]]

    for col in MUSE_EEG_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _parse_meta_from_path(filepath: str) -> Dict[str, str]:
    """
    Infer subject_id and trial_id from the file path when they are
    not embedded in the JSON itself.

    Expected structure:
        …/<subject_id>/<trial_id>.json
    """
    parts = filepath.replace("\\", "/").split("/")
    meta  = {}
    if len(parts) >= 2:
        meta[SUBJECT_COL] = parts[-2]
        meta[TRIAL_COL]   = os.path.splitext(parts[-1])[0]
    return meta


# ===========================================================================
# MAIN DATA LOADING — RAW MODE
# ===========================================================================

def _load_raw(data_root: str, config: dict) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load raw MUSE JSON files → (X_windows, y_labels, subject_ids, trial_ids)."""
    fs           = float(config.get("fs",             DEFAULT_FS))
    win_sec      = float(config.get("win_sec",        DEFAULT_WIN_SEC))
    overlap      = float(config.get("overlap",        DEFAULT_OVERLAP))
    hsi_thresh   = int(  config.get("hsi_threshold",  2))
    quality_filt = bool( config.get("quality_filter", True))
    label_map    = config.get("label_map", None)
    baseline_kw  = config.get("baseline_keyword", "baseline")

    pattern = os.path.join(data_root, "**", "*.json")
    files   = sorted(glob.glob(pattern, recursive=True))
    files   = [f for f in files
               if baseline_kw.lower() not in os.path.basename(f).lower()]

    if not files:
        raise FileNotFoundError(
            f"No JSON files found under '{data_root}' "
            f"(excluding files containing '{baseline_kw}')."
        )

    print(f"[load_raw] Found {len(files)} recording file(s).")

    all_windows, all_labels, all_subjects, all_trials = [], [], [], []

    for fp in files:
        df = _load_single_json(fp, config)
        if df is None:
            continue

        meta = _parse_meta_from_path(fp)
        if SUBJECT_COL not in df.columns:
            df[SUBJECT_COL] = meta.get(SUBJECT_COL, "unknown")
        if TRIAL_COL not in df.columns:
            df[TRIAL_COL] = meta.get(TRIAL_COL, "unknown")

        subject_id = str(df[SUBJECT_COL].iloc[0])
        trial_id   = str(df[TRIAL_COL].iloc[0])

        if LABEL_COL in df.columns:
            raw_label = df[LABEL_COL].iloc[0]
        else:
            raw_label = meta.get(TRIAL_COL, "unknown")

        label = label_map.get(str(raw_label), -1) if label_map is not None else raw_label

        if quality_filt:
            df = _apply_quality_filter(df, hsi_threshold=hsi_thresh)

        if df.empty:
            print(f"  [skip] '{fp}' — empty after quality filter.")
            continue

        eeg_cols = [c for c in MUSE_EEG_COLS if c in df.columns]
        if not eeg_cols:
            print(f"  [skip] '{fp}' — no EEG columns found.")
            continue

        signal  = np.stack(
            [_interp_nan(_to_num(df[c].values)) for c in eeg_cols], axis=-1
        )
        windows = _make_windows(signal, fs, win_sec, overlap)

        if windows.shape[0] == 0:
            print(f"  [skip] '{fp}' — signal too short for even one window.")
            continue

        n_win = windows.shape[0]
        all_windows.append(windows)
        all_labels.extend([label]      * n_win)
        all_subjects.extend([subject_id] * n_win)
        all_trials.extend([trial_id]   * n_win)

        print(f"  loaded '{os.path.basename(fp)}'  "
              f"subject={subject_id}  label={label}  windows={n_win}")

    if not all_windows:
        raise ValueError("No windows could be extracted from any file.")

    X = np.concatenate(all_windows, axis=0)
    y = np.array(all_labels)
    s = np.array(all_subjects)
    t = np.array(all_trials)

    print(f"\n[load_raw] Total windows : {X.shape[0]}")
    print(f"           Window shape  : {X.shape[1:]}")
    print(f"           Label dist    : {Counter(y.tolist())}")

    return X, y, s, t


# ===========================================================================
# MAIN DATA LOADING — CLEANED MODE
# ===========================================================================

def _load_cleaned(data_root: str, config: dict) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load pre-processed CSV/JSON files → (X_features, y_labels,
    subject_ids, trial_ids).

    Expected CSV schema
    -------------------
    Columns: subject_id, trial_id, label, feat_0, feat_1, … feat_N
    OR:       subject_id, trial_id, label, RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10

    If raw EEG columns are present, windowing + feature extraction are applied.
    """
    fs        = float(config.get("fs",      DEFAULT_FS))
    win_sec   = float(config.get("win_sec", DEFAULT_WIN_SEC))
    overlap   = float(config.get("overlap", DEFAULT_OVERLAP))
    label_map = config.get("label_map", None)
    baseline_kw = config.get("baseline_keyword", "baseline")

    csv_files  = sorted(glob.glob(os.path.join(data_root, "**", "*.csv"),  recursive=True))
    json_files = sorted(glob.glob(os.path.join(data_root, "**", "*.json"), recursive=True))
    files = csv_files + json_files
    files = [f for f in files
             if baseline_kw.lower() not in os.path.basename(f).lower()]

    if not files:
        raise FileNotFoundError(f"No CSV/JSON files found under '{data_root}'.")

    print(f"[load_cleaned] Found {len(files)} file(s).")

    all_X, all_y, all_s, all_t = [], [], [], []

    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(fp)
        else:
            with open(fp, "r") as fh:
                raw = json.load(fh)
            df = pd.DataFrame(raw if isinstance(raw, list) else raw.get("data", raw))

        meta       = _parse_meta_from_path(fp)
        subject_id = str(df[SUBJECT_COL].iloc[0]) if SUBJECT_COL in df.columns \
                     else meta.get(SUBJECT_COL, "unknown")
        trial_id   = str(df[TRIAL_COL].iloc[0])   if TRIAL_COL   in df.columns \
                     else meta.get(TRIAL_COL, "unknown")
        raw_label  = df[LABEL_COL].iloc[0]         if LABEL_COL   in df.columns \
                     else meta.get(TRIAL_COL, "unknown")
        label      = label_map.get(str(raw_label), -1) if label_map else raw_label

        drop_cols = [SUBJECT_COL, TRIAL_COL, LABEL_COL,
                     "HeadBandOn", "TimeStamp"] + MUSE_HSI_COLS
        feat_df   = df.drop(columns=[c for c in drop_cols if c in df.columns])
        eeg_cols  = [c for c in MUSE_EEG_COLS if c in feat_df.columns]

        if eeg_cols:
            signal  = np.stack(
                [_interp_nan(_to_num(feat_df[c].values)) for c in eeg_cols], axis=-1
            )
            windows = _make_windows(signal, fs, win_sec, overlap)
            if windows.shape[0] == 0:
                print(f"  [skip] '{fp}' — signal too short.")
                continue
            X_file = extract_eeg_features(windows, config, fs=fs)
            n_win  = X_file.shape[0]
        else:
            X_file = feat_df.apply(pd.to_numeric, errors="coerce").fillna(0).values
            X_file = X_file.astype(np.float32)
            n_win  = X_file.shape[0]

        all_X.append(X_file)
        all_y.extend([label]      * n_win)
        all_s.extend([subject_id] * n_win)
        all_t.extend([trial_id]   * n_win)

        print(f"  loaded '{os.path.basename(fp)}'  "
              f"subject={subject_id}  label={label}  rows={n_win}")

    if not all_X:
        raise ValueError("No data could be loaded from any file.")

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y)
    s = np.array(all_s)
    t = np.array(all_t)

    print(f"\n[load_cleaned] Total samples : {X.shape[0]}")
    print(f"               Feature dim   : {X.shape[1]}")
    print(f"               Label dist    : {Counter(y.tolist())}")

    return X, y, s, t


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

def load_eeg_data(
    data_root: str,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified EEG data loader for the Emognition dataset.

    Parameters
    ----------
    data_root : Path to the dataset root directory.
    config    : Configuration dictionary.  Key options:

        mode             (str)   'raw' or 'cleaned'   [default: 'raw']
        fs               (float) Sampling rate in Hz   [default: 256.0]
        win_sec          (float) Window length (s)     [default: 4.0]
        overlap          (float) Window overlap [0,1)  [default: 0.5]
        quality_filter   (bool)  Apply HSI filter      [default: True]  raw only
        hsi_threshold    (int)   Max HSI value (1–4)   [default: 2]     raw only
        baseline_keyword (str)   Keyword to skip
                                 baseline files        [default: 'baseline']
        label_map        (dict)  Map string→int labels [default: None]

    Returns
    -------
    X            : raw mode  → (N, W, C) windows
                   cleaned   → (N, F)    feature matrix
    y_labels     : (N,) label array
    subject_ids  : (N,) subject identifier array
    trial_ids    : (N,) trial identifier array

    Notes
    -----
    * Baseline reduction is intentionally NOT performed here.
      Use ``baseline_reduction.py`` before calling this function if needed.
    * In 'raw' mode call ``extract_eeg_features(X, config)`` to get features.
    """
    mode = config.get("mode", "raw").strip().lower()
    if mode == "raw":
        return _load_raw(data_root, config)
    elif mode == "cleaned":
        return _load_cleaned(data_root, config)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'raw' or 'cleaned'.")


# ===========================================================================
# DATA SPLITTING
# ===========================================================================

def create_data_splits(
    y_labels:    np.ndarray,
    subject_ids: np.ndarray,
    config:      dict,
    trial_ids:   Optional[np.ndarray] = None,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Create train / val / test boolean index splits.

    Parameters
    ----------
    y_labels    : (N,) label array.
    subject_ids : (N,) subject identifier array.
    config      : Config dict.  Key option:

        split_strategy (str)
            'subject_independent' — whole subjects in one split only.
            'clip_independent'    — whole trials in one split only.
            'random'              — shuffled randomly (default).

    trial_ids   : (N,) trial identifiers (required for clip_independent).
    train_ratio : Fraction for training.
    val_ratio   : Fraction for validation.
    test_ratio  : Fraction for test.

    Returns
    -------
    Dict with keys 'train', 'val', 'test' — each a bool array of shape (N,).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    strategy  = config.get("split_strategy", "random").strip().lower()
    rng       = np.random.default_rng(config.get("seed", 42))
    N         = len(y_labels)
    train_idx = np.zeros(N, dtype=bool)
    val_idx   = np.zeros(N, dtype=bool)
    test_idx  = np.zeros(N, dtype=bool)

    if strategy == "subject_independent":
        subjects = np.unique(subject_ids)
        rng.shuffle(subjects)
        n_train    = max(1, int(len(subjects) * train_ratio))
        n_val      = max(1, int(len(subjects) * val_ratio))
        train_subs = set(subjects[:n_train])
        val_subs   = set(subjects[n_train : n_train + n_val])
        test_subs  = set(subjects[n_train + n_val :])
        train_idx  = np.isin(subject_ids, list(train_subs))
        val_idx    = np.isin(subject_ids, list(val_subs))
        test_idx   = np.isin(subject_ids, list(test_subs))
        print(f"[split] subject_independent — "
              f"train subs={len(train_subs)}  val subs={len(val_subs)}  "
              f"test subs={len(test_subs)}")

    elif strategy == "clip_independent":
        if trial_ids is None:
            raise ValueError(
                "'clip_independent' strategy requires trial_ids to be provided."
            )
        trials       = np.unique(trial_ids)
        rng.shuffle(trials)
        n_train      = max(1, int(len(trials) * train_ratio))
        n_val        = max(1, int(len(trials) * val_ratio))
        train_trials = set(trials[:n_train])
        val_trials   = set(trials[n_train : n_train + n_val])
        test_trials  = set(trials[n_train + n_val :])
        train_idx    = np.isin(trial_ids, list(train_trials))
        val_idx      = np.isin(trial_ids, list(val_trials))
        test_idx     = np.isin(trial_ids, list(test_trials))
        print(f"[split] clip_independent — "
              f"train clips={len(train_trials)}  val clips={len(val_trials)}  "
              f"test clips={len(test_trials)}")

    else:  # random
        indices = np.arange(N)
        rng.shuffle(indices)
        n_train = int(N * train_ratio)
        n_val   = int(N * val_ratio)
        train_idx[indices[:n_train]]               = True
        val_idx  [indices[n_train:n_train + n_val]] = True
        test_idx [indices[n_train + n_val:]]        = True
        print(f"[split] random — train={train_idx.sum()}  "
              f"val={val_idx.sum()}  test={test_idx.sum()}")

    return {"train": train_idx, "val": val_idx, "test": test_idx}
