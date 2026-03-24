"""
EEG Data Loader Module — Emognition Dataset
============================================================
Loads MUSE EEG JSON files, applies quality filtering,
optional baseline reduction (InvBase), windowing, and
feature extraction.

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import os
import glob
import json
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional external feature extractor
# ---------------------------------------------------------------------------
try:
    from eeg_feature_extractor import extract_eeg_features
    _HAS_EXTRACTOR = True
except ImportError:
    _HAS_EXTRACTOR = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EEG_COLS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
HSI_COLS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def _to_num(x) -> np.ndarray:
    """Convert various input types to a float64 numpy array."""
    if isinstance(x, list):
        if not x:
            return np.array([], dtype=np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64)
    return np.asarray([x], dtype=np.float64)


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN / Inf values in a 1-D array."""
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def check_json_structure(data_root: str, num_samples: int = 3) -> None:
    """Print the top-level structure of a few JSON files — useful for debugging."""
    files = sorted(glob.glob(os.path.join(data_root, "**", "*.json"), recursive=True))[:num_samples]
    if not files:
        print(f"[check_json_structure] No JSON files found under '{data_root}'.")
        return
    for fp in files:
        with open(fp, "r") as fh:
            data = json.load(fh)
        print(f"\n--- {fp} ---")
        if isinstance(data, dict):
            for k, v in data.items():
                vlen = len(v) if hasattr(v, "__len__") else "scalar"
                print(f"  key='{k}'  type={type(v).__name__}  len={vlen}")
        elif isinstance(data, list):
            print(f"  list of {len(data)} items; first item: "
                  f"{list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
        else:
            print(f"  type={type(data)}")


# ===========================================================================
# BASELINE REDUCTION  (InvBase)
# ===========================================================================

def apply_baseline_reduction(
    signal: np.ndarray,
    baseline: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply InvBase baseline reduction.

    Divides the trial FFT by the baseline FFT per channel, then converts
    back to the time domain.

    Parameters
    ----------
    signal   : (T, C) trial signal.
    baseline : (T, C) resting-state baseline (same length as signal).
    eps      : Small constant to prevent division by zero.

    Returns
    -------
    reduced  : (T, C) float32 array.
    """
    fft_trial    = np.fft.rfft(signal,   axis=0)
    fft_baseline = np.fft.rfft(baseline, axis=0)
    fft_reduced  = fft_trial / (np.abs(fft_baseline) + eps)
    reduced      = np.fft.irfft(fft_reduced, n=len(signal), axis=0)
    return reduced.astype(np.float32)


def _load_baseline_files(files: list, data_root: str) -> dict:
    """
    Load per-subject baseline recordings.

    Parameters
    ----------
    files     : All recording file paths (used to discover subjects).
    data_root : Dataset root directory.

    Returns
    -------
    baseline_dict : { subject_id : (T, 4) float32 array }
    """
    baseline_dict = {}
    print("   Loading baseline recordings...")

    for fpath in files:
        fname   = os.path.basename(fpath)
        parts   = fname.split("_")
        if len(parts) < 2:
            continue
        subject = parts[0]
        if subject in baseline_dict or "BASELINE" in fname.upper():
            continue

        candidates = [
            os.path.join(data_root, f"{subject}_BASELINE_STIMULUS_MUSE.json"),
            os.path.join(data_root, subject,
                         f"{subject}_BASELINE_STIMULUS_MUSE.json"),
            os.path.join(data_root, subject,
                         f"{subject}_BASELINE_STIMULUS_MUSE_cleaned",
                         f"{subject}_BASELINE_STIMULUS_MUSE_cleaned.json"),
        ]

        for bp in candidates:
            if not os.path.exists(bp):
                continue
            try:
                with open(bp, "r") as fh:
                    data = json.load(fh)
                channels = [
                    _interp_nan(_to_num(data.get(c, [])))
                    for c in EEG_COLS
                ]
                L = min(len(ch) for ch in channels)
                if L == 0:
                    break
                sig = np.stack([ch[:L] for ch in channels], axis=1).astype(np.float32)
                sig -= np.nanmean(sig, axis=0, keepdims=True)
                baseline_dict[subject] = sig
            except Exception as exc:
                print(f"   [warning] baseline load failed for {subject}: {exc}")
            break  # stop after first match

    print(f"   Loaded {len(baseline_dict)} baseline recording(s).")
    return baseline_dict


# ===========================================================================
# MAIN DATA LOADER
# ===========================================================================

def load_eeg_data(
    data_root: str,
    config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load MUSE EEG recordings from *data_root*.

    Parameters
    ----------
    data_root : Path to the dataset root directory.
    config    : Config object **or** dict.  Expected attributes / keys:

        EEG_FS                  (float) Sampling rate in Hz        [256.0]
        EEG_WINDOW_SEC          (float) Window length in seconds   [4.0]
        EEG_OVERLAP             (float) Window overlap [0, 1)      [0.5]
        USE_BASELINE_REDUCTION  (bool)  Apply InvBase reduction    [False]
        SUPERCLASS_MAP          (dict)  emotion_str → class_str
        MODE                    (str)   'raw' | 'cleaned'          ['cleaned']
        SUBJECT_INDEPENDENT     (bool)  (used downstream)
        CLIP_INDEPENDENT        (bool)  (used downstream)

    Returns
    -------
    X_raw       : (N, W, 4)  raw EEG windows  float32
    y_labels    : (N,)       integer class labels
    subject_ids : (N,)       subject identifier strings
    trial_ids   : (N,)       trial identifier strings  (<subject>_<emotion>)
    label_to_id : dict       class_name → integer
    """

    def _cfg(key, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    fs             = float(_cfg("EEG_FS",                 256.0))
    win_sec        = float(_cfg("EEG_WINDOW_SEC",         4.0))
    overlap        = float(_cfg("EEG_OVERLAP",            0.5))
    use_baseline   = bool( _cfg("USE_BASELINE_REDUCTION", False))
    superclass_map = _cfg("SUPERCLASS_MAP", {})
    mode           = str(  _cfg("MODE", "cleaned")).lower()

    win_samples  = int(win_sec * fs)
    step_samples = max(1, int(win_samples * (1.0 - overlap)))

    print("\n" + "=" * 70)
    print("  LOADING EEG DATA  (MUSE)")
    print("=" * 70)
    print(f"  Root            : {data_root}")
    print(f"  Mode            : {mode}")
    print(f"  Fs              : {fs} Hz")
    print(f"  Window          : {win_sec}s  ({win_samples} samples)")
    print(f"  Overlap         : {overlap}")
    print(f"  Baseline reduc. : {use_baseline}")

    # ------------------------------------------------------------------
    # Discover files — raw or cleaned
    # ------------------------------------------------------------------
    if mode == "raw":
        patterns = [
            os.path.join(data_root, "*_STIMULUS_MUSE.json"),
            os.path.join(data_root, "*", "*_STIMULUS_MUSE.json"),
            os.path.join(data_root, "*", "*_STIMULUS_MUSE",
                         "*_STIMULUS_MUSE.json"),
        ]
    else:  # cleaned
        patterns = [
            os.path.join(data_root, "*_STIMULUS_MUSE_cleaned.json"),
            os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned.json"),
            os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned",
                         "*_STIMULUS_MUSE_cleaned.json"),
        ]

    files = sorted({p for pat in patterns for p in glob.glob(pat)})

    # Exclude baseline files from the main file list
    files = [f for f in files if "BASELINE" not in os.path.basename(f).upper()]

    print(f"\n  Found {len(files)} MUSE {mode} file(s).")

    if not files:
        raise FileNotFoundError(
            f"No MUSE {mode} files found under '{data_root}'.\n"
            f"  Raw    pattern : *_STIMULUS_MUSE.json\n"
            f"  Cleaned pattern: *_STIMULUS_MUSE_cleaned.json\n"
            f"Run check_json_structure('{data_root}') to inspect the layout."
        )

    # ------------------------------------------------------------------
    # Load baseline recordings (optional)
    # ------------------------------------------------------------------
    baseline_dict = {}
    if use_baseline:
        baseline_dict = _load_baseline_files(files, data_root)

    # ------------------------------------------------------------------
    # Process each file
    # ------------------------------------------------------------------
    all_windows, all_labels, all_subjects, all_trials = [], [], [], []
    skipped       = Counter()
    reduced_count = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.split("_")

        if len(parts) < 2:
            skipped["parse_error"] += 1
            continue

        subject = parts[0]
        emotion = parts[1].upper()

        if emotion not in superclass_map:
            skipped["unknown_emotion"] += 1
            continue

        superclass = superclass_map[emotion]
        trial_id   = f"{subject}_{emotion}"

        try:
            with open(fpath, "r") as fh:
                data = json.load(fh)

            # Extract & interpolate channels
            channels = [_interp_nan(_to_num(data.get(c, []))) for c in EEG_COLS]
            L = min(len(ch) for ch in channels)
            if L == 0:
                skipped["no_data"] += 1
                continue

            # Quality mask (applied to both raw and cleaned)
            hsi_arrays = [_to_num(data.get(h, []))[:L] for h in HSI_COLS]
            head_on    = _to_num(data.get("HeadBandOn", []))[:L]

            mask = np.ones(L, dtype=bool)
            if len(head_on) == L:
                mask &= (head_on == 1)
            for hsi in hsi_arrays:
                if len(hsi) == L:
                    mask &= np.isfinite(hsi) & (hsi <= 2)
            for ch in channels:
                mask &= np.isfinite(ch[:L])

            channels = [ch[:L][mask] for ch in channels]
            L = len(channels[0])
            if L < win_samples:
                skipped["insufficient_length"] += 1
                continue

            # Stack → (T, 4), zero-mean per channel
            signal = np.stack(channels, axis=1).astype(np.float32)
            signal -= np.nanmean(signal, axis=0, keepdims=True)

            # Baseline reduction
            if use_baseline and subject in baseline_dict:
                baseline_sig = baseline_dict[subject]
                common       = min(len(signal), len(baseline_sig))
                signal       = apply_baseline_reduction(
                    signal[:common], baseline_sig[:common]
                )
                L            = len(signal)
                reduced_count += 1

            # Windowing
            n_before = len(all_windows)
            for start in range(0, L - win_samples + 1, step_samples):
                all_windows.append(signal[start : start + win_samples])
                all_labels.append(superclass)
                all_subjects.append(subject)
                all_trials.append(trial_id)

            if len(all_windows) == n_before:
                skipped["insufficient_length"] += 1

        except Exception as exc:
            skipped["parse_error"] += 1
            print(f"  [warning] {fname}: {exc}")
            continue

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  Windows extracted : {len(all_windows)}")
    if skipped:
        print(f"  Skipped           : {dict(skipped)}")
    if use_baseline:
        print(f"  Baseline applied  : {reduced_count} file(s)")

    if not all_windows:
        raise ValueError("No valid EEG windows could be extracted.")

    # ------------------------------------------------------------------
    # Assemble outputs
    # ------------------------------------------------------------------
    X_raw         = np.stack(all_windows).astype(np.float32)   # (N, W, 4)
    unique_labels = sorted(set(all_labels))
    label_to_id   = {lab: i for i, lab in enumerate(unique_labels)}
    y_labels      = np.array([label_to_id[l] for l in all_labels], dtype=np.int64)
    subject_ids   = np.array(all_subjects)
    trial_ids     = np.array(all_trials)

    print(f"\n  X shape           : {X_raw.shape}")
    print(f"  Label distribution: {Counter(all_labels)}")
    print(f"  label_to_id       : {label_to_id}")
    print("=" * 70 + "\n")

    return X_raw, y_labels, subject_ids, trial_ids, label_to_id


# ===========================================================================
# DATA SPLITTING
# ===========================================================================

def create_data_splits(
    y_labels:    np.ndarray,
    subject_ids: np.ndarray,
    config,
    trial_ids:   Optional[np.ndarray] = None,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Create train / val / test index arrays.

    Parameters
    ----------
    y_labels    : (N,) label array.
    subject_ids : (N,) subject identifier array.
    config      : Config object or dict.  Key options:

        SUBJECT_INDEPENDENT (bool) — whole subjects in one split only.
        CLIP_INDEPENDENT    (bool) — whole trials in one split only.
        (default)                  — random shuffle.

    trial_ids   : (N,) trial IDs (required for CLIP_INDEPENDENT).
    train_ratio : Fraction for training.
    val_ratio   : Fraction for validation.
    test_ratio  : Fraction for test.

    Returns
    -------
    Dict with keys 'train', 'val', 'test' — each an integer index array.
    """
    def _cfg(key, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    subject_independent = bool(_cfg("SUBJECT_INDEPENDENT", False))
    clip_independent    = bool(_cfg("CLIP_INDEPENDENT",    False))
    seed                = int( _cfg("SEED", _cfg("seed", 42)))

    rng = np.random.default_rng(seed)
    N   = len(y_labels)

    print("\n" + "=" * 70)
    print("  CREATING DATA SPLITS")
    print("=" * 70)

    if subject_independent:
        print("  Strategy : SUBJECT-INDEPENDENT")
        subjects = np.unique(subject_ids)
        rng.shuffle(subjects)
        n_test  = max(1, int(len(subjects) * test_ratio))
        n_val   = max(1, int(len(subjects) * val_ratio))
        test_s  = subjects[:n_test]
        val_s   = subjects[n_test : n_test + n_val]
        train_s = subjects[n_test + n_val :]
        train_mask = np.isin(subject_ids, train_s)
        val_mask   = np.isin(subject_ids, val_s)
        test_mask  = np.isin(subject_ids, test_s)
        print(f"  Train subjects : {len(train_s)}  "
              f"Val : {len(val_s)}  Test : {len(test_s)}")

    elif clip_independent and trial_ids is not None:
        print("  Strategy : CLIP-INDEPENDENT")
        trials = np.unique(trial_ids)
        rng.shuffle(trials)
        n_test   = max(1, int(len(trials) * test_ratio))
        n_val    = max(1, int(len(trials) * val_ratio))
        test_t   = trials[:n_test]
        val_t    = trials[n_test : n_test + n_val]
        train_t  = trials[n_test + n_val :]
        train_mask = np.isin(trial_ids, train_t)
        val_mask   = np.isin(trial_ids, val_t)
        test_mask  = np.isin(trial_ids, test_t)
        print(f"  Train trials : {len(train_t)}  "
              f"Val : {len(val_t)}  Test : {len(test_t)}")

    else:
        print("  Strategy : RANDOM")
        indices = np.arange(N)
        rng.shuffle(indices)
        n_test  = int(N * test_ratio)
        n_val   = int(N * val_ratio)
        train_mask = np.zeros(N, dtype=bool)
        val_mask   = np.zeros(N, dtype=bool)
        test_mask  = np.zeros(N, dtype=bool)
        test_mask [indices[:n_test]]               = True
        val_mask  [indices[n_test : n_test + n_val]] = True
        train_mask[indices[n_test + n_val :]]       = True

    split_indices = {
        "train": np.where(train_mask)[0],
        "val":   np.where(val_mask)[0],
        "test":  np.where(test_mask)[0],
    }

    print(f"\n  Train samples : {len(split_indices['train'])}")
    print(f"  Val   samples : {len(split_indices['val'])}")
    print(f"  Test  samples : {len(split_indices['test'])})")
    for split, idx in split_indices.items():
        print(f"  {split:5s} class dist : {dict(Counter(y_labels[idx].tolist()))}")
    print("=" * 70 + "\n")

    return split_indices
