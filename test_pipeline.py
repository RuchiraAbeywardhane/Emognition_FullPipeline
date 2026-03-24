"""
test_pipeline.py — Real Data Loading Test
============================================================
Loads the actual Emognition dataset using data_loader.py
and prints a summary of what was loaded.

Usage
-----
    python test_pipeline.py --data "path/to/your/dataset" --mode raw
    python test_pipeline.py --data "path/to/your/dataset" --mode cleaned
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import CFG, LOADER_CFG
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
        "--mode", default="raw", choices=["raw", "cleaned"],
        help="Dataset mode: 'raw' for MUSE JSON files, 'cleaned' for pre-processed CSV/JSON.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Emognition — Data Loading Test")
    print("=" * 60)
    print(f"  Dataset path : {args.data}")
    print(f"  Mode         : {args.mode}")
    print("=" * 60 + "\n")

    # Build config from CFG, override mode with the one passed by user
    cfg = {**LOADER_CFG, "mode": args.mode}

    # ----------------------------------------------------------------
    # Step 1 — Load data
    # ----------------------------------------------------------------
    print(">>> Loading data ...\n")
    X, y, subject_ids, trial_ids = load_eeg_data(args.data, cfg)

    # ----------------------------------------------------------------
    # Step 2 — Print summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LOAD SUMMARY")
    print("=" * 60)
    print(f"  X shape        : {X.shape}")
    print(f"  Total windows  : {X.shape[0]}")
    print(f"  Subjects       : {sorted(set(subject_ids.tolist()))}")
    print(f"  Trials         : {sorted(set(trial_ids.tolist()))}")
    print(f"  Labels (int)   : {sorted(set(y.tolist()))}")

    label_map_inv = {v: k for k, v in CFG["data"]["label_map"].items()}
    label_counts  = {label_map_inv.get(lbl, lbl): int((y == lbl).sum())
                     for lbl in sorted(set(y.tolist()))}
    print(f"  Label counts   : {label_counts}")
    print(f"  NaN in X       : {np.isnan(X).any()}")
    print(f"  Inf in X       : {np.isinf(X).any()}")

    # ----------------------------------------------------------------
    # Step 3 — Create splits
    # ----------------------------------------------------------------
    print("\n>>> Creating data splits ...\n")
    splits = create_data_splits(y, subject_ids, cfg, trial_ids=trial_ids)

    print("\n" + "=" * 60)
    print("  SPLIT SUMMARY")
    print("=" * 60)
    for name, mask in splits.items():
        print(f"  {name:5s} : {mask.sum()} samples  "
              f"subjects={sorted(set(subject_ids[mask].tolist()))}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
