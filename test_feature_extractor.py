"""
test_feature_extractor.py — Feature Extractor Test
============================================================
Loads real EEG data, runs the feature extractor, and prints
a detailed report of the output.

Usage
-----
    python test_feature_extractor.py --data "path/to/dataset" --mode raw
    python test_feature_extractor.py --data "path/to/dataset" --mode cleaned
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import CONFIG
from data_loader import load_eeg_data
from eeg_feature_extractor import extract_eeg_features, get_feature_names, BANDS, CHANNEL_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Test the EEG feature extractor on real data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", metavar="PATH", required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--mode", default="cleaned", choices=["raw", "cleaned"],
        help="Dataset mode.",
    )
    args = parser.parse_args()

    CONFIG.MODE = args.mode

    # ----------------------------------------------------------------
    # Step 1 — Load raw windows from dataset
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Step 1 : Loading raw EEG windows")
    print("=" * 70)

    X_raw, y, subject_ids, trial_ids, label_to_id = load_eeg_data(args.data, CONFIG)

    N, W, C = X_raw.shape
    print(f"  X_raw shape : {X_raw.shape}  (windows × samples × channels)")
    print(f"  Labels      : {label_to_id}")

    # ----------------------------------------------------------------
    # Step 2 — Run feature extraction
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Step 2 : Extracting features")
    print("=" * 70)

    X_feat = extract_eeg_features(X_raw, CONFIG)

    print(f"  X_feat shape : {X_feat.shape}  (windows × features)")
    print(f"  Expected     : ({N}, {C * 26})")

    # ----------------------------------------------------------------
    # Step 3 — Shape and integrity checks
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Step 3 : Integrity checks")
    print("=" * 70)

    checks = {
        "Output is 2-D"                    : X_feat.ndim == 2,
        "Row count matches window count"   : X_feat.shape[0] == N,
        f"Feature count == {C} × 26 = {C*26}": X_feat.shape[1] == C * 26,
        "No NaN values"                    : not np.isnan(X_feat).any(),
        "No Inf values"                    : not np.isinf(X_feat).any(),
        "dtype is float32"                 : X_feat.dtype == np.float32,
    }

    all_passed = True
    for desc, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_passed = False
        print(f"  [{status}]  {desc}")

    # ----------------------------------------------------------------
    # Step 4 — Feature names
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Step 4 : Feature names")
    print("=" * 70)

    names = get_feature_names(n_channels=C)
    print(f"  Total feature names : {len(names)}")
    print(f"  First 10            : {names[:10]}")
    print(f"  Last  10            : {names[-10:]}")

    name_check = len(names) == X_feat.shape[1]
    print(f"\n  [{'PASS' if name_check else 'FAIL'}]  "
          f"Name count matches feature count ({len(names)} == {X_feat.shape[1]})")
    if not name_check:
        all_passed = False

    # ----------------------------------------------------------------
    # Step 5 — Per-feature statistics (mean ± std across all windows)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Step 5 : Per-feature statistics  (first window, all features)")
    print("=" * 70)

    feat_means = X_feat.mean(axis=0)   # (F,)
    feat_stds  = X_feat.std(axis=0)    # (F,)
    feat_mins  = X_feat.min(axis=0)
    feat_maxs  = X_feat.max(axis=0)

    print(f"  {'Feature':<30}  {'Mean':>12}  {'Std':>12}  {'Min':>12}  {'Max':>12}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    # Print stats for each band PSD and DE feature of the first channel
    # plus all temporal features of the first channel (26 features total)
    first_ch_indices = list(range(26))
    for i in first_ch_indices:
        print(f"  {names[i]:<30}  "
              f"{feat_means[i]:>12.4f}  "
              f"{feat_stds[i]:>12.4f}  "
              f"{feat_mins[i]:>12.4f}  "
              f"{feat_maxs[i]:>12.4f}")

    # ----------------------------------------------------------------
    # Step 6 — Per-band PSD sanity check
    #          Alpha power should be positive for resting EEG
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Step 6 : Band power sanity check  (mean across all windows & channels)")
    print("=" * 70)

    band_names = list(BANDS.keys())
    print(f"  {'Band':<10}  {'Mean PSD':>14}  {'Mean DE':>14}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}")

    for b_idx, band in enumerate(band_names):
        # PSD indices: features 0–4 per channel; average across all channels
        psd_cols = [ch * 26 + b_idx       for ch in range(C)]
        de_cols  = [ch * 26 + 5 + b_idx   for ch in range(C)]
        mean_psd = X_feat[:, psd_cols].mean()
        mean_de  = X_feat[:, de_cols ].mean()
        print(f"  {band:<10}  {mean_psd:>14.4f}  {mean_de:>14.4f}")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Input  X_raw  : {X_raw.shape}")
    print(f"  Output X_feat : {X_feat.shape}")
    print(f"  Features/window : {C} channels × 26 = {C * 26}")
    print(f"  Overall result  : {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
