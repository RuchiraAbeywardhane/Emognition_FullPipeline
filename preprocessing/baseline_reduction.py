"""
Baseline Reduction Module — Emognition Dataset
============================================================
Dedicated script for applying baseline reduction to raw EEG
recordings before they are passed to the data loader or any
downstream model.

Method
------
  InvBase (Inverse Baseline):
      reduced(t) = signal(t) / (baseline_mean + eps)

  This is the standard method used in Emognition-style pipelines.
  It normalises each channel relative to the subject's resting-state
  recording, suppressing inter-subject amplitude differences.

Typical Usage
-------------
  1. Run this script once to produce baseline-reduced files on disk.
  2. Point data_loader.py at the reduced output directory.

  OR call apply_baseline_reduction() directly in your pipeline:

      from baseline_reduction import load_baseline_files, apply_baseline_reduction

      baselines   = load_baseline_files(baseline_files, data_root)
      signal_reduced = apply_baseline_reduction(signal, baselines[subject_id])

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import os
import glob
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column name constants (must match data_loader.py)
# ---------------------------------------------------------------------------
MUSE_EEG_COLS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
MUSE_HSI_COLS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]
SUBJECT_COL   = "subject_id"
TRIAL_COL     = "trial_id"
LABEL_COL     = "label"


# ===========================================================================
# UTILITY HELPERS
# ===========================================================================

def _to_num(x) -> np.ndarray:
    """Convert various input types to a float64 numpy array."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64)
    return np.array(x, dtype=np.float64)


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D array."""
    a = a.copy()
    nans = np.isnan(a)
    if not nans.any():
        return a
    idx = np.arange(len(a))
    a[nans] = np.interp(idx[nans], idx[~nans], a[~nans])
    return a


def _load_json_to_df(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load a MUSE JSON file into a DataFrame.
    Handles three common formats:
      1. { "data": [ {sample}, … ] }
      2. [ {sample}, … ]
      3. { "col": [values], … }  (column-oriented)
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
        print(f"  [warning] Unrecognised format in '{filepath}' — skipping.")
        return None

    for col in MUSE_EEG_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ===========================================================================
# BASELINE REDUCTION
# ===========================================================================

def apply_baseline_reduction(
    signal:   np.ndarray,
    baseline: np.ndarray,
    eps:      float = 1e-12,
    method:   str   = "invbase",
) -> np.ndarray:
    """
    Apply baseline reduction to a continuous EEG signal.

    Parameters
    ----------
    signal   : Array of shape (T, C) — the trial recording.
    baseline : Array of shape (B, C) or (C,) — the baseline recording or
               its pre-computed per-channel mean.
    eps      : Small constant for numerical stability.
    method   : Reduction method.  Supported values:

        'invbase'  (default)
            reduced(t) = signal(t) / (mean(baseline) + eps)
            Divides each sample by the baseline mean — suppresses
            inter-subject amplitude differences.

        'zscore'
            reduced(t) = (signal(t) - mean(baseline)) /
                         (std(baseline) + eps)
            Z-scores the signal relative to the baseline distribution.

        'subtract'
            reduced(t) = signal(t) - mean(baseline)
            Simple mean subtraction.

    Returns
    -------
    reduced : Array of shape (T, C), same dtype as *signal*.
    """
    signal   = _to_num(signal)           # (T, C)
    baseline = _to_num(baseline)

    # If a full baseline recording is passed, compute statistics
    if baseline.ndim == 2:
        base_mean = np.mean(baseline, axis=0)   # (C,)
        base_std  = np.std( baseline, axis=0)   # (C,)
    else:
        # Already a mean vector
        base_mean = baseline
        base_std  = np.ones_like(base_mean)

    method = method.lower().strip()

    if method == "invbase":
        reduced = signal / (base_mean + eps)
    elif method == "zscore":
        reduced = (signal - base_mean) / (base_std + eps)
    elif method == "subtract":
        reduced = signal - base_mean
    else:
        raise ValueError(
            f"Unknown baseline method '{method}'. "
            "Choose 'invbase', 'zscore', or 'subtract'."
        )

    return reduced.astype(signal.dtype)


# ===========================================================================
# BASELINE FILE LOADING
# ===========================================================================

def load_baseline_files(
    files:     List[str],
    data_root: str,
    method:    str = "invbase",
) -> Dict[str, np.ndarray]:
    """
    Load baseline JSON recordings for all subjects and compute the
    per-channel summary statistic needed for reduction.

    Parameters
    ----------
    files     : List of absolute or relative paths to baseline JSON files.
                Filenames are expected to follow the pattern
                ``<subject_id>_baseline.json`` or be stored under a
                subject folder:  ``<data_root>/<subject_id>/baseline.json``.
    data_root : Root directory (used to resolve relative paths).
    method    : Baseline reduction method (see apply_baseline_reduction).
                Determines what statistic is stored ('invbase' / 'subtract'
                store the mean; 'zscore' stores a (mean, std) tuple).

    Returns
    -------
    baselines : Dict mapping subject_id (str) → baseline array.
                  'invbase' / 'subtract' → shape (C,)   per-channel mean
                  'zscore'               → shape (2, C)  [mean, std]
    """
    baselines: Dict[str, np.ndarray] = {}

    for fp in files:
        if not os.path.isabs(fp):
            fp = os.path.join(data_root, fp)

        if not os.path.exists(fp):
            print(f"  [warning] Baseline file not found: '{fp}' — skipping.")
            continue

        df = _load_json_to_df(fp)
        if df is None:
            continue

        # Determine subject_id
        if SUBJECT_COL in df.columns:
            subject_id = str(df[SUBJECT_COL].iloc[0])
        else:
            # Infer from path: …/<subject_id>/baseline.json
            #                  or …/<subject_id>_baseline.json
            parts = fp.replace("\\", "/").split("/")
            fname = os.path.splitext(parts[-1])[0]           # e.g. "S01_baseline"
            subject_id = fname.replace("_baseline", "").replace("baseline_", "")
            if subject_id == "baseline" and len(parts) >= 2:
                subject_id = parts[-2]                        # parent folder name

        eeg_cols = [c for c in MUSE_EEG_COLS if c in df.columns]
        if not eeg_cols:
            print(f"  [warning] No EEG columns in '{fp}' — skipping.")
            continue

        signal = np.stack(
            [_interp_nan(_to_num(df[c].values)) for c in eeg_cols],
            axis=-1,
        )  # (T, C)

        base_mean = np.mean(signal, axis=0)   # (C,)
        base_std  = np.std( signal, axis=0)   # (C,)

        if method.lower() == "zscore":
            baselines[subject_id] = np.stack([base_mean, base_std], axis=0)  # (2, C)
        else:
            baselines[subject_id] = base_mean  # (C,)

        print(f"  [baseline] subject='{subject_id}'  "
              f"channels={len(eeg_cols)}  samples={len(signal)}")

    print(f"\n[load_baseline_files] Loaded baselines for "
          f"{len(baselines)} subject(s).")
    return baselines


def discover_baseline_files(
    data_root:       str,
    baseline_keyword: str = "baseline",
) -> List[str]:
    """
    Recursively find all baseline JSON files under *data_root*.

    Parameters
    ----------
    data_root        : Root directory to search.
    baseline_keyword : Files whose name contains this keyword (case-insensitive)
                       are treated as baselines.

    Returns
    -------
    List of absolute file paths.
    """
    pattern = os.path.join(data_root, "**", "*.json")
    all_files = glob.glob(pattern, recursive=True)
    baseline_files = [
        f for f in all_files
        if baseline_keyword.lower() in os.path.basename(f).lower()
    ]
    print(f"[discover_baseline_files] Found {len(baseline_files)} "
          f"baseline file(s) under '{data_root}'.")
    return sorted(baseline_files)


# ===========================================================================
# BATCH PROCESSING — reduce an entire directory and save to disk
# ===========================================================================

def reduce_dataset(
    data_root:        str,
    output_root:      str,
    method:           str  = "invbase",
    baseline_keyword: str  = "baseline",
    eps:              float = 1e-12,
    overwrite:        bool  = False,
) -> None:
    """
    Apply baseline reduction to every recording in *data_root* and
    write the reduced files to *output_root*, preserving the original
    directory structure.

    Baseline files are detected automatically via *baseline_keyword*.
    Each subject's own baseline is used when available; a global mean
    across all subjects is used as a fallback.

    Parameters
    ----------
    data_root        : Input directory containing raw JSON files.
    output_root      : Output directory for reduced JSON files.
    method           : Baseline reduction method ('invbase', 'zscore',
                       'subtract').
    baseline_keyword : Filename keyword identifying baseline recordings.
    eps              : Numerical stability constant.
    overwrite        : If False, skip files that already exist in
                       output_root.
    """
    # --- Discover and load all baselines first ---
    baseline_files = discover_baseline_files(data_root, baseline_keyword)
    if not baseline_files:
        raise FileNotFoundError(
            f"No baseline files found under '{data_root}' "
            f"(keyword='{baseline_keyword}')."
        )

    baselines = load_baseline_files(baseline_files, data_root, method=method)

    # Compute global fallback baseline (mean of all subjects)
    all_means = np.stack(
        [v if v.ndim == 1 else v[0] for v in baselines.values()],
        axis=0,
    )  # (S, C)
    global_baseline_mean = np.mean(all_means, axis=0)  # (C,)
    if method == "zscore":
        all_stds = np.stack(
            [v[1] for v in baselines.values() if v.ndim == 2],
            axis=0,
        )
        global_baseline_std = np.mean(all_stds, axis=0)
        global_baseline = np.stack(
            [global_baseline_mean, global_baseline_std], axis=0
        )
    else:
        global_baseline = global_baseline_mean

    print(f"\n[reduce_dataset] Starting batch reduction "
          f"(method='{method}') …")

    # --- Discover all non-baseline recording files ---
    pattern   = os.path.join(data_root, "**", "*.json")
    all_files = sorted(glob.glob(pattern, recursive=True))
    rec_files = [
        f for f in all_files
        if baseline_keyword.lower() not in os.path.basename(f).lower()
    ]

    print(f"[reduce_dataset] {len(rec_files)} recording file(s) to process.")

    for fp in rec_files:
        # Mirror directory structure under output_root
        rel_path    = os.path.relpath(fp, data_root)
        out_path    = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if not overwrite and os.path.exists(out_path):
            print(f"  [skip] Already exists: '{out_path}'")
            continue

        df = _load_json_to_df(fp)
        if df is None:
            continue

        # Resolve subject_id to pick correct baseline
        if SUBJECT_COL in df.columns:
            subject_id = str(df[SUBJECT_COL].iloc[0])
        else:
            parts      = fp.replace("\\", "/").split("/")
            subject_id = parts[-2] if len(parts) >= 2 else "unknown"

        baseline = baselines.get(subject_id, global_baseline)
        if subject_id not in baselines:
            print(f"  [fallback] No baseline for subject '{subject_id}' "
                  f"— using global mean.")

        eeg_cols = [c for c in MUSE_EEG_COLS if c in df.columns]
        if not eeg_cols:
            print(f"  [skip] No EEG columns in '{fp}'.")
            continue

        signal = np.stack(
            [_interp_nan(_to_num(df[c].values)) for c in eeg_cols],
            axis=-1,
        )  # (T, C)

        reduced = apply_baseline_reduction(signal, baseline, eps=eps, method=method)

        # Write reduced EEG columns back into the DataFrame
        for i, col in enumerate(eeg_cols):
            df[col] = reduced[:, i]

        # Save as JSON (preserves metadata columns)
        df.to_json(out_path, orient="records", indent=2)
        print(f"  reduced → '{out_path}'")

    print(f"\n[reduce_dataset] Done.  Output written to '{output_root}'.")


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply baseline reduction to the Emognition raw dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Path to the raw dataset directory.",
    )
    parser.add_argument(
        "--output_root", required=True,
        help="Path to write the baseline-reduced files.",
    )
    parser.add_argument(
        "--method", default="invbase",
        choices=["invbase", "zscore", "subtract"],
        help="Baseline reduction method.",
    )
    parser.add_argument(
        "--baseline_keyword", default="baseline",
        help="Filename keyword that identifies baseline recordings.",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-12,
        help="Small constant for numerical stability.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reduce_dataset(
        data_root        = args.data_root,
        output_root      = args.output_root,
        method           = args.method,
        baseline_keyword = args.baseline_keyword,
        eps              = args.eps,
        overwrite        = args.overwrite,
    )
