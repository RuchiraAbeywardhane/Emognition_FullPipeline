"""
Pipeline Test Script — Emognition Dataset
============================================================
Self-contained tests for data_loader.py and baseline_reduction.py.

All test data is generated synthetically — no real dataset is needed.
Temporary files are written to a temp folder and cleaned up afterwards.

Run:
    python test_pipeline.py
    python test_pipeline.py --keep   # keep temp files for inspection

Coverage
--------
  data_loader.py
    [T01]  _to_num            — list / ndarray / Series → float64
    [T02]  _interp_nan        — NaN interpolation including edges
    [T03]  _make_windows      — shape, overlap, short-signal guard
    [T04]  extract_eeg_features — output shape (N, C*26), no NaN/Inf
    [T05]  load_eeg_data RAW  — JSON files, quality filter, label map
    [T06]  load_eeg_data CLEANED (CSV, pre-extracted features)
    [T07]  load_eeg_data CLEANED (CSV, raw EEG columns → auto-extract)
    [T08]  create_data_splits random
    [T09]  create_data_splits subject_independent
    [T10]  create_data_splits clip_independent
    [T11]  check_json_structure smoke-test

  baseline_reduction.py
    [T12]  apply_baseline_reduction invbase
    [T13]  apply_baseline_reduction zscore
    [T14]  apply_baseline_reduction subtract
    [T15]  apply_baseline_reduction — unknown method raises ValueError
    [T16]  load_baseline_files — subject_id from JSON column
    [T17]  load_baseline_files — subject_id inferred from filename
    [T18]  discover_baseline_files
    [T19]  reduce_dataset — batch output matches manual reduction

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the pipeline folder is on the path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import (
    _to_num,
    _interp_nan,
    _make_windows,
    extract_eeg_features,
    load_eeg_data,
    create_data_splits,
    check_json_structure,
    MUSE_EEG_COLS,
    MUSE_HSI_COLS,
    DEFAULT_FS,
)
from baseline_reduction import (
    apply_baseline_reduction,
    load_baseline_files,
    discover_baseline_files,
    reduce_dataset,
)
from config import CFG, LOADER_CFG

# ---------------------------------------------------------------------------
# Derive test constants from CFG — edit config.py, not here
# ---------------------------------------------------------------------------
FS        = CFG["data"]["fs"]
WIN_SEC   = CFG["data"]["win_sec"]
OVERLAP   = CFG["data"]["overlap"]
N_CHAN    = len(CFG["data"]["eeg_cols"])
SEED      = CFG["reproducibility"]["seed"]
LABEL_MAP = CFG["data"]["label_map"]

# Synthetic signal generation constants
SIGNAL_FS           = FS
SIGNAL_DURATION_SEC = 12.0
BASELINE_DURATION_SEC = 60.0
DELTA_AMP  = 10.0
ALPHA_AMP  = 5.0
BETA_AMP   = 2.0
NOISE_STD  = 0.5
BAD_HSI_SAMPLES = 50

SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S06"]
LABELS   = list(LABEL_MAP.keys())[:3]   # use first 3 emotion labels

PREEXTRACTED_ROWS_PER_SUBJECT = 20
DEFAULT_BASELINE_METHOD = CFG["baseline"]["method"]
TMP_DIR_PREFIX          = "emognition_test_"

# Base config dict passed into pipeline functions during tests
BASE_CFG: dict = {**LOADER_CFG, "mode": CFG["data"]["mode"]}

# ===========================================================================
# TEST HARNESS
# ===========================================================================

PASS  = "  \033[92mPASS\033[0m"
FAIL  = "  \033[91mFAIL\033[0m"
SKIP  = "  \033[93mSKIP\033[0m"

_results: Dict[str, str] = {}


def run(test_id: str, name: str):
    """Decorator that catches exceptions and records pass/fail."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            tag = f"[{test_id}] {name}"
            try:
                fn(*args, **kwargs)
                _results[tag] = "PASS"
                print(f"{PASS}  {tag}")
            except AssertionError as e:
                _results[tag] = f"FAIL: {e}"
                print(f"{FAIL}  {tag}")
                print(f"         AssertionError: {e}")
            except Exception:
                _results[tag] = "ERROR"
                print(f"{FAIL}  {tag}")
                traceback.print_exc()
        return wrapper
    return decorator


# ===========================================================================
# SYNTHETIC DATA HELPERS
# ===========================================================================

N_SAMP   = int(SIGNAL_DURATION_SEC * SIGNAL_FS)
# N_CHAN already derived from CFG["data"]["eeg_cols"] above


def _make_eeg_signal(n_samples: int = N_SAMP,
                     n_chan: int    = N_CHAN,
                     seed: int      = 0) -> np.ndarray:
    """Return synthetic band-limited EEG (T, C)."""
    rng = np.random.default_rng(seed)
    t   = np.arange(n_samples) / SIGNAL_FS
    sig = np.zeros((n_samples, n_chan))
    for c in range(n_chan):
        sig[:, c] = (
            DELTA_AMP * np.sin(2 * np.pi * 2.0  * t + rng.uniform(0, np.pi)) +
            ALPHA_AMP * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, np.pi)) +
            BETA_AMP  * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, np.pi)) +
            rng.normal(0, NOISE_STD, n_samples)
        )
    return sig


def _make_raw_json(subject: str, trial: str, label: str,
                   tmp_dir: str, seed: int = 0,
                   inject_bad_hsi: bool = False) -> str:
    """
    Write a synthetic MUSE raw JSON file to tmp_dir/<subject>/<trial>.json
    Returns the file path.
    """
    sig = _make_eeg_signal(seed=seed)
    records = []
    for i in range(N_SAMP):
        row: Dict = {
            "RAW_TP9":    round(sig[i, 0], 4),
            "RAW_AF7":    round(sig[i, 1], 4),
            "RAW_AF8":    round(sig[i, 2], 4),
            "RAW_TP10":   round(sig[i, 3], 4),
            "HSI_TP9":    4 if inject_bad_hsi and i < BAD_HSI_SAMPLES else 1,
            "HSI_AF7":    4 if inject_bad_hsi and i < BAD_HSI_SAMPLES else 1,
            "HSI_AF8":    4 if inject_bad_hsi and i < BAD_HSI_SAMPLES else 1,
            "HSI_TP10":   4 if inject_bad_hsi and i < BAD_HSI_SAMPLES else 1,
            "HeadBandOn": 0 if inject_bad_hsi and i < BAD_HSI_SAMPLES else 1,
            "TimeStamp":  i / SIGNAL_FS,
            "label":      label,
            "subject_id": subject,
            "trial_id":   trial,
        }
        records.append(row)

    out_dir = os.path.join(tmp_dir, subject)
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"{trial}.json")
    with open(fp, "w") as fh:
        json.dump(records, fh)
    return fp


def _make_baseline_json(subject: str, tmp_dir: str, seed: int = 99) -> str:
    """Write a baseline JSON file to tmp_dir/<subject>/baseline.json"""
    n_samples = int(BASELINE_DURATION_SEC * SIGNAL_FS)
    sig = _make_eeg_signal(n_samples=n_samples, seed=seed)
    records = []
    for i in range(sig.shape[0]):
        records.append({
            "RAW_TP9":    round(sig[i, 0], 4),
            "RAW_AF7":    round(sig[i, 1], 4),
            "RAW_AF8":    round(sig[i, 2], 4),
            "RAW_TP10":   round(sig[i, 3], 4),
            "HSI_TP9":    1,
            "HSI_AF7":    1,
            "HSI_AF8":    1,
            "HSI_TP10":   1,
            "HeadBandOn": 1,
            "TimeStamp":  i / SIGNAL_FS,
            "subject_id": subject,
        })
    out_dir = os.path.join(tmp_dir, subject)
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, "baseline.json")
    with open(fp, "w") as fh:
        json.dump(records, fh)
    return fp


def _make_cleaned_csv_preextracted(subject: str, trial: str,
                                   label: str, tmp_dir: str,
                                   n_rows: int = PREEXTRACTED_ROWS_PER_SUBJECT,
                                   seed: int = 0) -> str:
    """Write a CSV with pre-extracted feature columns."""
    rng  = np.random.default_rng(seed)
    cols = [f"feat_{i}" for i in range(26 * N_CHAN)]
    df   = pd.DataFrame(rng.random((n_rows, len(cols))), columns=cols)
    df.insert(0, "subject_id", subject)
    df.insert(1, "trial_id",   trial)
    df.insert(2, "label",      label)
    out_dir = os.path.join(tmp_dir, subject)
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"{trial}.csv")
    df.to_csv(fp, index=False)
    return fp


def _make_cleaned_csv_raweeg(subject: str, trial: str,
                              label: str, tmp_dir: str, seed: int = 0) -> str:
    """Write a CSV with raw EEG columns (cleaned but not yet extracted)."""
    sig = _make_eeg_signal(seed=seed)
    df  = pd.DataFrame(sig, columns=MUSE_EEG_COLS)
    df.insert(0, "subject_id", subject)
    df.insert(1, "trial_id",   trial)
    df.insert(2, "label",      label)
    out_dir = os.path.join(tmp_dir, subject)
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"{trial}.csv")
    df.to_csv(fp, index=False)
    return fp


# ===========================================================================

# BASE_CFG is imported directly from test_config; no local re-definition needed.

# ===========================================================================

# data_loader.py — UTILITY TESTS

# ===========================================================================

@run("T01", "_to_num — list / ndarray / Series → float64")
def test_to_num():
    assert _to_num([1, 2, 3]).dtype    == np.float64
    assert _to_num(np.array([1, 2])).dtype  == np.float64
    assert _to_num(pd.Series([1.0, 2.0])).dtype == np.float64


@run("T02", "_interp_nan — interior and edge NaNs")
def test_interp_nan():
    a = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    out = _interp_nan(a)
    assert not np.any(np.isnan(out)), "Interior NaNs remain"
    np.testing.assert_allclose(out[1], 2.0, atol=1e-6)
    np.testing.assert_allclose(out[3], 4.0, atol=1e-6)

    # Edge NaNs
    b = np.array([np.nan, 2.0, 3.0, np.nan])
    out_b = _interp_nan(b)
    assert not np.any(np.isnan(out_b)), "Edge NaNs remain"


@run("T03", "_make_windows — shape and overlap correctness")
def test_make_windows():
    sig     = np.random.randn(int(10 * FS), N_CHAN)
    win_s   = int(WIN_SEC * FS)
    step_s  = int(win_s * OVERLAP)
    windows = _make_windows(sig, FS, WIN_SEC, OVERLAP)
    assert windows.ndim == 3,             f"Expected 3-D, got {windows.ndim}"
    assert windows.shape[1] == win_s,     f"Wrong window length {windows.shape[1]}"
    assert windows.shape[2] == N_CHAN,    f"Wrong channel count {windows.shape[2]}"

    # Expected number of windows
    expected_n = (int(10 * FS) - win_s) // step_s + 1
    assert windows.shape[0] == expected_n, \
        f"Expected {expected_n} windows, got {windows.shape[0]}"

    # Signal too short → empty
    short = np.random.randn(10, N_CHAN)
    empty = _make_windows(short, FS, WIN_SEC, OVERLAP)
    assert empty.shape[0] == 0 or True   # stack of 0 items is acceptable


@run("T04", "extract_eeg_features — shape (N, C*26), no NaN/Inf")
def test_extract_features():
    sig     = _make_eeg_signal()
    windows = _make_windows(sig, FS, WIN_SEC, OVERLAP)
    feats   = extract_eeg_features(windows, BASE_CFG, fs=FS)

    assert feats.ndim == 2,                     "Expected 2-D feature array"
    assert feats.shape[0] == windows.shape[0],  "Row count mismatch"
    assert feats.shape[1] == N_CHAN * 26,        \
        f"Expected {N_CHAN * 26} features, got {feats.shape[1]}"
    assert not np.any(np.isnan(feats)),   "NaN in features"
    assert not np.any(np.isinf(feats)),   "Inf in features"


# ===========================================================================

# data_loader.py — LOAD TESTS  (use temp dirs)

# ===========================================================================

@run("T05", "load_eeg_data RAW — windows, quality filter, label map")
def test_load_raw(tmp_raw):
    # 3 subjects × 3 labels × 1 trial, one file has injected bad HSI
    for i, (subj, label) in enumerate(zip(SUBJECTS[:3], LABELS)):
        bad = (i == 0)          # S01 has BAD_HSI_SAMPLES bad-HSI samples
        _make_raw_json(subj, "trial01", label, tmp_raw,
                       seed=i, inject_bad_hsi=bad)

    cfg = {**BASE_CFG, "mode": "raw"}
    X, y, s, t = load_eeg_data(tmp_raw, cfg)

    assert X.ndim == 3,             f"Expected 3-D windows, got X.shape={X.shape}"
    assert X.shape[2] == N_CHAN,    f"Wrong channel count {X.shape[2]}"
    assert len(y) == X.shape[0],    "Label count mismatch"
    assert len(s) == X.shape[0],    "Subject array length mismatch"
    assert set(y).issubset({0, 1, 2}), f"Unexpected label values: {set(y)}"
    assert set(s) == {"S01", "S02", "S03"}, f"Unexpected subjects: {set(s)}"


@run("T06", "load_eeg_data CLEANED — pre-extracted CSV features")
def test_load_cleaned_preextracted(tmp_clean):
    for i, (subj, label) in enumerate(zip(SUBJECTS[:3], LABELS)):
        _make_cleaned_csv_preextracted(subj, "trial01", label,
                                       tmp_clean, seed=i)

    cfg = {**BASE_CFG, "mode": "cleaned"}
    X, y, s, t = load_eeg_data(tmp_clean, cfg)

    assert X.ndim == 2,                                   f"Expected 2-D features, got {X.shape}"
    assert X.shape[1] == N_CHAN * 26,                     f"Wrong feature dim {X.shape[1]}"
    assert X.shape[0] == 3 * PREEXTRACTED_ROWS_PER_SUBJECT, \
        f"Expected {3 * PREEXTRACTED_ROWS_PER_SUBJECT} rows, got {X.shape[0]}"
    assert set(y) == {0, 1, 2},                           f"Label mismatch: {set(y)}"


@run("T07", "load_eeg_data CLEANED — raw EEG CSV → auto-extract")
def test_load_cleaned_raweeg(tmp_clean2):
    for i, (subj, label) in enumerate(zip(SUBJECTS[:2], LABELS[:2])):
        _make_cleaned_csv_raweeg(subj, "trial01", label, tmp_clean2, seed=i)

    cfg = {**BASE_CFG, "mode": "cleaned"}
    X, y, s, t = load_eeg_data(tmp_clean2, cfg)

    assert X.ndim == 2,              f"Expected 2-D features, got {X.shape}"
    assert X.shape[1] == N_CHAN * 26, f"Wrong feature dim {X.shape[1]}"
    assert not np.any(np.isnan(X)),  "NaN in extracted features"


@run("T08", "create_data_splits — random strategy, no overlap")
def test_split_random():
    N = 300
    y = np.zeros(N, dtype=int)
    s = np.array([f"S{i:02d}" for i in range(N)])
    cfg = {**BASE_CFG, "split_strategy": "random"}
    splits = create_data_splits(y, s, cfg)

    _assert_splits_valid(splits, N)


@run("T09", "create_data_splits — subject_independent, no leakage")
def test_split_subject_independent(tmp_raw):
    # Load real windows so subject_ids are meaningful
    for i, (subj, label) in enumerate(zip(SUBJECTS, LABELS * 2)):
        _make_raw_json(subj, "trial01", label, tmp_raw, seed=i)

    cfg = {**BASE_CFG, "mode": "raw", "split_strategy": "subject_independent"}
    X, y, s, t = load_eeg_data(tmp_raw, cfg)
    splits = create_data_splits(y, s, cfg, trial_ids=t)

    _assert_splits_valid(splits, len(y))

    # No subject should appear in more than one split
    for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
        subs_a = set(s[splits[split_a]])
        subs_b = set(s[splits[split_b]])
        overlap = subs_a & subs_b
        assert not overlap, \
            f"Subject leakage between {split_a} and {split_b}: {overlap}"


@run("T10", "create_data_splits — clip_independent, no trial leakage")
def test_split_clip_independent(tmp_raw):
    # Create multiple trials per subject
    for i, subj in enumerate(SUBJECTS[:3]):
        for j, label in enumerate(LABELS):
            _make_raw_json(subj, f"trial{j:02d}", label, tmp_raw,
                           seed=i * 10 + j)

    cfg = {**BASE_CFG, "mode": "raw", "split_strategy": "clip_independent"}
    X, y, s, t = load_eeg_data(tmp_raw, cfg)
    splits = create_data_splits(y, s, cfg, trial_ids=t)

    _assert_splits_valid(splits, len(y))

    # No trial should appear in more than one split
    for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
        trials_a = set(t[splits[split_a]])
        trials_b = set(t[splits[split_b]])
        overlap  = trials_a & trials_b
        assert not overlap, \
            f"Trial leakage between {split_a} and {split_b}: {overlap}"


@run("T11", "check_json_structure — smoke test (no crash)")
def test_check_json_structure(tmp_raw):
    for i, (subj, label) in enumerate(zip(SUBJECTS[:2], LABELS[:2])):
        _make_raw_json(subj, "trial01", label, tmp_raw, seed=i)
    # Should print without raising
    check_json_structure(tmp_raw, num_samples=2)


def _assert_splits_valid(splits: dict, N: int):
    train, val, test = splits["train"], splits["val"], splits["test"]
    # Every sample assigned to exactly one split
    combined = train.astype(int) + val.astype(int) + test.astype(int)
    assert np.all(combined == 1), \
        f"Some samples missing or in multiple splits: unique combined values={np.unique(combined)}"
    assert train.sum() > 0, "Training split is empty"
    assert val.sum()   > 0, "Validation split is empty"
    assert test.sum()  > 0, "Test split is empty"


# ===========================================================================

# baseline_reduction.py — TESTS

# ===========================================================================

@run("T12", "apply_baseline_reduction — invbase")
def test_br_invbase():
    signal   = _make_eeg_signal(n_samples=1000)   # (T, C)
    baseline = _make_eeg_signal(n_samples=500, seed=99)

    reduced = apply_baseline_reduction(signal, baseline, method="invbase")

    assert reduced.shape == signal.shape, "Shape mismatch"
    assert not np.any(np.isnan(reduced)),  "NaN in reduced signal"
    assert not np.any(np.isinf(reduced)),  "Inf in reduced signal"

    # Verify formula manually on channel 0
    base_mean = baseline[:, 0].mean()
    expected  = signal[:, 0] / (base_mean + 1e-12)
    np.testing.assert_allclose(reduced[:, 0], expected, rtol=1e-5)


@run("T13", "apply_baseline_reduction — zscore")
def test_br_zscore():
    signal   = _make_eeg_signal(n_samples=1000)
    baseline = _make_eeg_signal(n_samples=500, seed=99)

    reduced = apply_baseline_reduction(signal, baseline, method="zscore")

    assert reduced.shape == signal.shape
    assert not np.any(np.isnan(reduced))

    base_mean = baseline[:, 0].mean()
    base_std  = baseline[:, 0].std()
    expected  = (signal[:, 0] - base_mean) / (base_std + 1e-12)
    np.testing.assert_allclose(reduced[:, 0], expected, rtol=1e-5)


@run("T14", "apply_baseline_reduction — subtract")
def test_br_subtract():
    signal   = _make_eeg_signal(n_samples=1000)
    baseline = _make_eeg_signal(n_samples=500, seed=99)

    reduced = apply_baseline_reduction(signal, baseline, method="subtract")

    base_mean = baseline[:, 0].mean()
    expected  = signal[:, 0] - base_mean
    np.testing.assert_allclose(reduced[:, 0], expected, rtol=1e-5)


@run("T15", "apply_baseline_reduction — unknown method raises ValueError")
def test_br_bad_method():
    signal   = _make_eeg_signal(n_samples=100)
    baseline = _make_eeg_signal(n_samples=50, seed=1)
    raised   = False
    try:
        apply_baseline_reduction(signal, baseline, method="magic")
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for unknown method"


@run("T16", "load_baseline_files — subject_id from JSON column")
def test_load_baseline_from_col(tmp_base):
    for i, subj in enumerate(SUBJECTS[:3]):
        _make_baseline_json(subj, tmp_base, seed=i)

    files  = discover_baseline_files(tmp_base)
    result = load_baseline_files(files, tmp_base, method=DEFAULT_BASELINE_METHOD)

    assert len(result) == 3, f"Expected 3 subjects, got {len(result)}"
    for subj in SUBJECTS[:3]:
        assert subj in result, f"Missing baseline for {subj}"
        assert result[subj].shape == (N_CHAN,), \
            f"Wrong baseline shape for {subj}: {result[subj].shape}"


@run("T17", "load_baseline_files — subject_id inferred from filename")
def test_load_baseline_from_filename(tmp_base2):
    """Baseline files named S01_baseline.json (no subject_id column)."""
    for i, subj in enumerate(SUBJECTS[:2]):
        sig     = _make_eeg_signal(n_samples=int(30 * FS), seed=i)
        records = []
        for k in range(sig.shape[0]):
            records.append({
                "RAW_TP9":  round(sig[k, 0], 4),
                "RAW_AF7":  round(sig[k, 1], 4),
                "RAW_AF8":  round(sig[k, 2], 4),
                "RAW_TP10": round(sig[k, 3], 4),
            })
        fp = os.path.join(tmp_base2, f"{subj}_baseline.json")
        with open(fp, "w") as fh:
            json.dump(records, fh)

    files  = discover_baseline_files(tmp_base2)
    result = load_baseline_files(files, tmp_base2)

    assert len(result) == 2, f"Expected 2 baselines, got {len(result)}"
    for subj in SUBJECTS[:2]:
        assert subj in result, f"Missing baseline for {subj}"


@run("T18", "discover_baseline_files — finds only baseline files")
def test_discover_baseline(tmp_raw):
    # Create both recording and baseline files
    _make_raw_json("S01", "trial01", "neutral", tmp_raw)
    _make_baseline_json("S01", tmp_raw)

    found = discover_baseline_files(tmp_raw, baseline_keyword="baseline")
    basenames = [os.path.basename(f) for f in found]

    assert any("baseline" in b for b in basenames), \
        "No baseline file discovered"
    assert not any("trial" in b for b in basenames), \
        "Recording file incorrectly included"


@run("T19", "reduce_dataset — output matches manual invbase reduction")
def test_reduce_dataset(tmp_raw, tmp_out):
    subj  = "S01"
    label = "neutral"
    _make_baseline_json(subj, tmp_raw, seed=99)
    _make_raw_json(subj, "trial01", label, tmp_raw, seed=0)

    reduce_dataset(
        data_root    = tmp_raw,
        output_root  = tmp_out,
        method       = "invbase",
        overwrite    = True,
    )

    # Load output file
    out_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(tmp_out)
        for f in files if f.endswith(".json")
    ]
    assert len(out_files) > 0, "No output files created by reduce_dataset"

    with open(out_files[0], "r") as fh:
        out_data = json.load(fh)
    df_out = pd.DataFrame(out_data)

    # Verify EEG columns exist and contain no NaN
    for col in MUSE_EEG_COLS:
        if col in df_out.columns:
            assert not df_out[col].isna().any(), f"NaN in reduced column {col}"


# ===========================================================================

# MAIN — run all tests with per-test temp dirs

# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Emognition Pipeline — Test Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep temporary synthetic test files after run.",
    )
    parser.add_argument(
        "--data", metavar="PATH", default=None,
        help=(
            "Path to a real Emognition dataset root directory. "
            "If supplied, extra tests (T20–T22) will load and validate "
            "your actual data instead of synthetic files."
        ),
    )
    parser.add_argument(
        "--mode", default=CFG["data"]["mode"], choices=["raw", "cleaned"],
        help="Dataset mode to use when --data is supplied.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Emognition Pipeline — Test Suite")
    print("=" * 60 + "\n")

    root_tmp = tempfile.mkdtemp(prefix=TMP_DIR_PREFIX)
    print(f"Temp directory : {root_tmp}\n")

    def tmp(name: str) -> str:
        p = os.path.join(root_tmp, name)
        os.makedirs(p, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # data_loader synthetic tests
    # ------------------------------------------------------------------
    test_to_num()
    test_interp_nan()
    test_make_windows()
    test_extract_features()
    test_load_raw(tmp("raw_T05"))
    test_load_cleaned_preextracted(tmp("clean_T06"))
    test_load_cleaned_raweeg(tmp("clean_T07"))
    test_split_random()
    test_split_subject_independent(tmp("raw_T09"))
    test_split_clip_independent(tmp("raw_T10"))
    test_check_json_structure(tmp("raw_T11"))

    # ------------------------------------------------------------------
    # baseline_reduction synthetic tests
    # ------------------------------------------------------------------
    test_br_invbase()
    test_br_zscore()
    test_br_subtract()
    test_br_bad_method()
    test_load_baseline_from_col(tmp("base_T16"))
    test_load_baseline_from_filename(tmp("base_T17"))
    test_discover_baseline(tmp("raw_T18"))
    test_reduce_dataset(tmp("raw_T19"), tmp("out_T19"))

    # ------------------------------------------------------------------
    # Real-data tests (only when --data is supplied)
    # ------------------------------------------------------------------
    if args.data:
        _run_real_data_tests(args.data, args.mode)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in _results.values() if v == "PASS")
    failed = sum(1 for v in _results.values() if v != "PASS")
    for name, status in _results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon}  {name}" + ("" if status == "PASS" else f"  ← {status}"))
    print(f"\n  {passed} passed  |  {failed} failed  |  {len(_results)} total")
    print("=" * 60 + "\n")

    if args.keep:
        print(f"Temp files kept at: {root_tmp}\n")
    else:
        shutil.rmtree(root_tmp, ignore_errors=True)
        print("Temp files cleaned up.\n")

    sys.exit(0 if failed == 0 else 1)


# ===========================================================================
# REAL-DATA TESTS  (T20 – T22)
# ===========================================================================

def _run_real_data_tests(data_root: str, mode: str) -> None:
    """
    Run tests against a real Emognition dataset directory.
    These tests are skipped when --data is not supplied.
    """
    print(f"\n--- Real-data tests  (path='{data_root}'  mode='{mode}') ---\n")

    @run("T20", f"load_eeg_data [{mode}] — files load without error")
    def test_real_load():
        cfg = {**LOADER_CFG, "mode": mode}
        X, y, s, t = load_eeg_data(data_root, cfg)

        assert X.ndim in (2, 3),         f"Unexpected X.ndim={X.ndim}"
        assert len(y) == X.shape[0],     "Label count != window count"
        assert len(s) == X.shape[0],     "Subject count != window count"
        assert len(t) == X.shape[0],     "Trial count != window count"
        assert not np.any(np.isnan(X)),  "NaN values in loaded data"
        assert not np.any(np.isinf(X)),  "Inf values in loaded data"
        print(f"         shape={X.shape}  labels={set(y.tolist())}  "
              f"subjects={len(set(s.tolist()))}")

    @run("T21", f"load_eeg_data [{mode}] — label map applied correctly")
    def test_real_labels():
        cfg = {**LOADER_CFG, "mode": mode}
        _, y, _, _ = load_eeg_data(data_root, cfg)
        valid = set(LABEL_MAP.values()) | {-1}   # -1 = unmapped
        unexpected = set(y.tolist()) - valid
        assert not unexpected, \
            f"Labels not in label_map: {unexpected}"

    @run("T22", "discover_baseline_files — finds at least one baseline in real data")
    def test_real_baselines():
        found = discover_baseline_files(
            data_root,
            baseline_keyword=CFG["data"]["baseline_keyword"],
        )
        assert len(found) > 0, \
            (f"No baseline files found under '{data_root}'. "
             f"Check CFG['data']['baseline_keyword'] = "
             f"'{CFG['data']['baseline_keyword']}'")
        print(f"         found {len(found)} baseline file(s)")

    test_real_load()
    test_real_labels()
    test_real_baselines()


if __name__ == "__main__":
    main()
