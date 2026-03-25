"""
test_feature_extractor.py — Feature Extractor Test
============================================================
Loads real EEG data then tests all three feature modes:
  1. statistical  — hand-crafted features (no PyTorch needed)
  2. deep         — CNN1D / LSTM / CNN_LSTM / Transformer embeddings
  3. combined     — statistical + deep concatenated

Usage
-----
    # Test with cleaned files (default)
    python test_feature_extractor.py --data "path/to/dataset" --mode cleaned

    # Test with raw files
    python test_feature_extractor.py --data "path/to/dataset" --mode raw

    # Test a specific deep extractor
    python test_feature_extractor.py --data "path/to/dataset" --mode raw --feature_mode deep --extractor cnn1d
    python test_feature_extractor.py --data "path/to/dataset" --mode raw --feature_mode deep --extractor lstm
    python test_feature_extractor.py --data "path/to/dataset" --mode raw --feature_mode deep --extractor cnn_lstm
    python test_feature_extractor.py --data "path/to/dataset" --mode raw --feature_mode deep --extractor transformer

    # Test combined mode
    python test_feature_extractor.py --data "path/to/dataset" --mode raw --feature_mode combined --extractor cnn1d

    # Test all deep extractors in one run
    python test_feature_extractor.py --data "path/to/dataset" --mode raw --test_all_deep
"""

from __future__ import annotations

import argparse
import sys
import os
import time

import numpy as np

# Add project root to path so 'config' and 'pipeline' are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.config import CONFIG
from data_loaders.data_loader import load_eeg_data
from feature_extraction.feature_extractor import (
    extract_eeg_features,
    get_feature_names,
    BANDS,
    CHANNEL_NAMES,
    clear_model_cache,
    _TORCH_AVAILABLE,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"

# ===========================================================================
# HELPERS
# ===========================================================================

def _check(desc: str, result: bool, results: list) -> None:
    status = PASS if result else FAIL
    print(f"{status}  {desc}")
    results.append((desc, result))


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ===========================================================================
# INDIVIDUAL TEST BLOCKS
# ===========================================================================

def test_statistical(X_raw: np.ndarray, results: list) -> np.ndarray:
    """Test the statistical feature extraction mode."""
    _section("Statistical Features")

    N, W, C = X_raw.shape
    CONFIG.FEATURE_MODE = "statistical"

    t0     = time.time()
    X_feat = extract_eeg_features(X_raw, CONFIG)
    elapsed = time.time() - t0

    expected_feats = C * 26
    names = get_feature_names(CONFIG, n_channels=C)

    print(f"  Input  : {X_raw.shape}")
    print(f"  Output : {X_feat.shape}   (expected ({N}, {expected_feats}))")
    print(f"  Time   : {elapsed:.2f}s")

    _check("Output is 2-D",                      X_feat.ndim == 2,                    results)
    _check("Row count matches window count",      X_feat.shape[0] == N,               results)
    _check(f"Feature count == {C}×26 = {expected_feats}", X_feat.shape[1] == expected_feats, results)
    _check("No NaN values",                       not np.isnan(X_feat).any(),          results)
    _check("No Inf values",                       not np.isinf(X_feat).any(),          results)
    _check("dtype is float32",                    X_feat.dtype == np.float32,          results)
    _check(f"Name count == {expected_feats}",     len(names) == expected_feats,        results)

    # Band power sanity — PSD must be positive
    band_names = list(BANDS.keys())
    print(f"\n  Band power check (mean across windows & channels):")
    print(f"  {'Band':<10}  {'Mean PSD':>14}  {'Mean DE':>14}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
    all_psd_positive = True
    for b_idx, band in enumerate(band_names):
        psd_cols = [ch * 26 + b_idx     for ch in range(C)]
        de_cols  = [ch * 26 + 5 + b_idx for ch in range(C)]
        mean_psd = float(X_feat[:, psd_cols].mean())
        mean_de  = float(X_feat[:, de_cols ].mean())
        print(f"  {band:<10}  {mean_psd:>14.4f}  {mean_de:>14.4f}")
        if mean_psd <= 0:
            all_psd_positive = False
    _check("All band PSD values > 0", all_psd_positive, results)

    # Per-feature stats table for channel TP9 (first 26 features)
    feat_means = X_feat.mean(axis=0)
    feat_stds  = X_feat.std(axis=0)
    print(f"\n  Per-feature stats — channel {CHANNEL_NAMES[0]} (first 26):")
    print(f"  {'Feature':<30}  {'Mean':>12}  {'Std':>12}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
    for i in range(26):
        print(f"  {names[i]:<30}  {feat_means[i]:>12.4f}  {feat_stds[i]:>12.4f}")

    return X_feat


def test_deep(X_raw: np.ndarray, extractor: str, embed_dim: int,
              results: list) -> np.ndarray:
    """Test a single deep extractor in 'deep' mode."""
    _section(f"Deep Features — {extractor.upper()}")

    if not _TORCH_AVAILABLE:
        print("  [SKIP]  PyTorch not installed — skipping deep tests.")
        print("          Install with:  pip install torch")
        return None

    N, W, C = X_raw.shape
    CONFIG.FEATURE_MODE   = "deep"
    CONFIG.DEEP_EXTRACTOR = extractor
    CONFIG.DEEP_EMBED_DIM = embed_dim

    print(f"  Input      : {X_raw.shape}")
    print(f"  Extractor  : {extractor}")
    print(f"  Embed dim  : {embed_dim}")
    print(f"  Device     : {CONFIG.DEVICE}")

    t0      = time.time()
    X_feat  = extract_eeg_features(X_raw, CONFIG)
    elapsed = time.time() - t0

    names = get_feature_names(CONFIG, n_channels=C)

    print(f"  Output     : {X_feat.shape}   (expected ({N}, {embed_dim}))")
    print(f"  Time       : {elapsed:.2f}s")

    _check("Output is 2-D",                 X_feat.ndim == 2,               results)
    _check("Row count matches window count", X_feat.shape[0] == N,          results)
    _check(f"Embed dim == {embed_dim}",      X_feat.shape[1] == embed_dim,  results)
    _check("No NaN values",                 not np.isnan(X_feat).any(),     results)
    _check("No Inf values",                 not np.isinf(X_feat).any(),     results)
    _check("dtype is float32",              X_feat.dtype == np.float32,     results)
    _check(f"Name count == {embed_dim}",    len(names) == embed_dim,        results)

    # Embedding statistics
    print(f"\n  Embedding stats (across all windows):")
    print(f"  Mean : {X_feat.mean():.6f}")
    print(f"  Std  : {X_feat.std():.6f}")
    print(f"  Min  : {X_feat.min():.6f}")
    print(f"  Max  : {X_feat.max():.6f}")

    clear_model_cache()
    return X_feat


def test_combined(X_raw: np.ndarray, extractor: str, embed_dim: int,
                  results: list) -> np.ndarray:
    """Test combined (statistical + deep) mode."""
    _section(f"Combined Features — statistical + {extractor.upper()}")

    if not _TORCH_AVAILABLE:
        print("  [SKIP]  PyTorch not installed — skipping combined test.")
        return None

    N, W, C = X_raw.shape
    CONFIG.FEATURE_MODE   = "combined"
    CONFIG.DEEP_EXTRACTOR = extractor
    CONFIG.DEEP_EMBED_DIM = embed_dim

    expected = C * 26 + embed_dim

    t0      = time.time()
    X_feat  = extract_eeg_features(X_raw, CONFIG)
    elapsed = time.time() - t0

    names = get_feature_names(CONFIG, n_channels=C)

    print(f"  Input    : {X_raw.shape}")
    print(f"  Output   : {X_feat.shape}   (expected ({N}, {expected}))")
    print(f"  Time     : {elapsed:.2f}s")

    _check("Output is 2-D",                  X_feat.ndim == 2,             results)
    _check("Row count matches window count",  X_feat.shape[0] == N,        results)
    _check(f"Feature count == {C}×26 + {embed_dim} = {expected}",
           X_feat.shape[1] == expected,                                     results)
    _check("No NaN values",                  not np.isnan(X_feat).any(),   results)
    _check("No Inf values",                  not np.isinf(X_feat).any(),   results)
    _check(f"Name count == {expected}",       len(names) == expected,      results)

    clear_model_cache()
    return X_feat


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test the EEG feature extractor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",         metavar="PATH", required=True)
    parser.add_argument("--mode",         default="cleaned", choices=["raw", "cleaned"],
                        help="Dataset file mode: 'raw' for *_STIMULUS_MUSE.json, "
                             "'cleaned' for *_STIMULUS_MUSE_cleaned.json")
    parser.add_argument("--feature_mode", default="statistical",
                        choices=["statistical", "deep", "combined"],
                        help="Feature extraction mode to test.")
    parser.add_argument("--extractor",    default="cnn1d",
                        choices=["cnn1d", "lstm", "cnn_lstm", "transformer"],
                        help="Deep extractor to use.")
    parser.add_argument("--embed_dim",    type=int, default=128,
                        help="Deep embedding output size.")
    parser.add_argument("--test_all_deep", action="store_true",
                        help="Run all four deep extractors one by one.")
    args = parser.parse_args()

    CONFIG.MODE = args.mode

    # ----------------------------------------------------------------
    # Print run configuration upfront
    # ----------------------------------------------------------------
    _section("Run Configuration")
    print(f"  Dataset path   : {args.data}")
    print(f"  Dataset mode   : {args.mode}  "
          f"({'*_STIMULUS_MUSE.json' if args.mode == 'raw' else '*_STIMULUS_MUSE_cleaned.json'})")
    print(f"  Feature mode   : {args.feature_mode}")
    print(f"  Deep extractor : {args.extractor}")
    print(f"  Embed dim      : {args.embed_dim}")
    print(f"  Test all deep  : {args.test_all_deep}")
    print(f"  PyTorch        : {'available' if _TORCH_AVAILABLE else 'NOT installed'}")

    # ----------------------------------------------------------------
    # Load data once — shared across all tests
    # ----------------------------------------------------------------
    _section("Step 1 : Loading EEG Windows")
    X_raw, y, subject_ids, trial_ids, label_to_id = load_eeg_data(args.data, CONFIG)
    N, W, C = X_raw.shape
    print(f"  X_raw shape    : {X_raw.shape}")
    print(f"  Labels         : {label_to_id}")
    print(f"  Subjects       : {sorted(set(subject_ids.tolist()))}")
    print(f"  Unique trials  : {len(np.unique(trial_ids))}")

    all_results = []

    # ----------------------------------------------------------------
    # Always run statistical
    # ----------------------------------------------------------------
    test_statistical(X_raw, all_results)

    # ----------------------------------------------------------------
    # Deep / combined tests
    # ----------------------------------------------------------------
    if args.test_all_deep:
        for ext in ["cnn1d", "lstm", "cnn_lstm", "transformer"]:
            test_deep(X_raw, ext, args.embed_dim, all_results)
        test_combined(X_raw, "cnn1d", args.embed_dim, all_results)

    elif args.feature_mode == "deep":
        test_deep(X_raw, args.extractor, args.embed_dim, all_results)

    elif args.feature_mode == "combined":
        test_deep(X_raw,     args.extractor, args.embed_dim, all_results)
        test_combined(X_raw, args.extractor, args.embed_dim, all_results)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    _section("FINAL SUMMARY")
    passed = sum(1 for _, r in all_results if r)
    failed = sum(1 for _, r in all_results if not r)

    for desc, result in all_results:
        icon = "✓" if result else "✗"
        print(f"  {icon}  {desc}")

    print(f"\n  Dataset mode    : {args.mode}")
    print(f"  Feature mode    : {args.feature_mode}")
    print(f"  {passed} passed  |  {failed} failed  |  {len(all_results)} total")
    print(f"  PyTorch available : {_TORCH_AVAILABLE}")
    print("=" * 70 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
