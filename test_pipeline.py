"""
test_pipeline.py — Real Data Loading Test
============================================================
Loads the actual Emognition dataset using data_loader.py
and prints a summary of what was loaded.

Usage
-----
    python test_pipeline.py --data "path/to/your/dataset" 
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
from data_loader import load_eeg_data, create_data_splits


def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect the Emognition dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", metavar="PATH", required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--mode", default="cleaned", choices=["raw", "cleaned"],
        help="'raw' for *_STIMULUS_MUSE.json, 'cleaned' for *_STIMULUS_MUSE_cleaned.json",
    )
    args = parser.parse_args()

    # Override MODE on the config object before passing to loader
    CONFIG.MODE = args.mode

    print("\n" + "=" * 70)
    print("  Emognition — Data Loading Test")
    print("=" * 70)
    print(f"  Dataset path : {args.data}")
    print(f"  Mode         : {args.mode}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1 — Load data
    # ----------------------------------------------------------------
    X, y, subject_ids, trial_ids, label_to_id = load_eeg_data(args.data, CONFIG)

    # ----------------------------------------------------------------
    # Step 2 — Print summary
    # ----------------------------------------------------------------
    id_to_label = {v: k for k, v in label_to_id.items()}
    label_counts = {
        id_to_label[lbl]: int((y == lbl).sum())
        for lbl in sorted(set(y.tolist()))
    }

    print("=" * 70)
    print("  LOAD SUMMARY")
    print("=" * 70)
    print(f"  X shape        : {X.shape}")
    print(f"  Total windows  : {X.shape[0]}")
    print(f"  Subjects       : {sorted(set(subject_ids.tolist()))}")
    print(f"  Unique trials  : {len(np.unique(trial_ids))}")
    print(f"  Label counts   : {label_counts}")
    print(f"  NaN in X       : {np.isnan(X).any()}")
    print(f"  Inf in X       : {np.isinf(X).any()}")

    # ----------------------------------------------------------------
    # Step 3 — Create splits
    # ----------------------------------------------------------------
    splits = create_data_splits(y, subject_ids, CONFIG, trial_ids=trial_ids)

    print("=" * 70)
    print("  SPLIT SUMMARY")
    print("=" * 70)
    for name, idx in splits.items():
        print(f"  {name:5s} : {len(idx):6d} samples  "
              f"subjects={sorted(set(subject_ids[idx].tolist()))}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
