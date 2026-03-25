"""
test_models.py — Model Test Script
============================================================
Tests every registered model using the real Emognition dataset.

What is tested per model
------------------------
  ✓ build_model() instantiation
  ✓ fit()  — trains without error
  ✓ predict()       — returns (N,) int array
  ✓ predict_proba() — returns (N, num_classes) float array summing to 1
  ✓ Output shapes correct
  ✓ No NaN / Inf in predictions
  ✓ get_params()    — returns a dict

Usage
-----
  # Test all models
  python tests/test_models.py --data "path/to/dataset"

  # Select specific models
  python tests/test_models.py --data "path/to/dataset" --models svm random_forest cnn1d

  # Deep models only
  python tests/test_models.py --data "path/to/dataset" --models mlp cnn1d lstm bilstm cnn_lstm transformer

  # Control training epochs for deep models
  python tests/test_models.py --data "path/to/dataset" --epochs 10

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

import argparse
import sys
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.config import CFG, CONFIG
from models import build_model, list_models, RAW_WINDOW_MODELS, FEATURE_MODELS
from feature_extraction.feature_extractor import extract_eeg_features
from data_loaders.data_loader import load_eeg_data, create_data_splits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PASS = "  [PASS]"
FAIL = "  [FAIL]"


# ===========================================================================
# HELPERS
# ===========================================================================

def _check(desc: str, result: bool, results: list) -> None:
    icon = PASS if result else FAIL
    print(f"{icon}  {desc}")
    results.append((desc, result))


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                       num_classes: int) -> np.ndarray:
    """Compute (num_classes, num_classes) confusion matrix without sklearn."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _print_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    """Pretty-print a confusion matrix."""
    n   = len(class_names)
    col_w = max(max(len(c) for c in class_names), 6)

    # Header
    header = "  " + " " * (col_w + 2)
    for c in class_names:
        header += f"  {c:^{col_w}}"
    print(header)
    print("  " + " " * (col_w + 2) + "  " + ("-" * col_w + "  ") * n)

    # Rows
    for i, row_name in enumerate(class_names):
        row = f"  {row_name:<{col_w}}  |"
        for j in range(n):
            row += f"  {cm[i, j]:^{col_w}}"
        print(row)

    # Footer: col totals
    print("  " + " " * (col_w + 2) + "  " + ("-" * col_w + "  ") * n)
    totals = "  " + "Total" + " " * (col_w - 3) + "  |"
    for j in range(n):
        totals += f"  {cm[:, j].sum():^{col_w}}"
    print(totals)


def _print_per_class_accuracy(cm: np.ndarray, class_names: list) -> list:
    """
    Print per-class accuracy (recall) from a confusion matrix.
    Returns list of per-class accuracies.
    """
    print(f"\n  {'Class':<14}  {'Correct':>8}  {'Total':>8}  {'Accuracy':>10}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*10}")
    per_class = []
    for i, name in enumerate(class_names):
        correct = int(cm[i, i])
        total   = int(cm[i].sum())
        acc     = correct / total if total > 0 else 0.0
        per_class.append(acc)
        bar = "█" * int(acc * 20)
        print(f"  {name:<14}  {correct:>8}  {total:>8}  {acc:>9.4f}  {bar}")
    return per_class


# ===========================================================================
# SINGLE MODEL TEST
# ===========================================================================

def test_one_model(
    name:        str,
    X_train:     np.ndarray, y_train: np.ndarray,
    X_val:       np.ndarray, y_val:   np.ndarray,
    X_test:      np.ndarray, y_test:  np.ndarray,
    num_classes: int,
    class_names: list,
    results:     list,
) -> dict:
    """Run the full test suite for one model. Returns a summary dict."""
    _section(f"Model : {name.upper()}")
    model_results: list = []
    summary = {"name": name, "passed": 0, "failed": 0,
               "train_time": 0.0, "test_acc": None,
               "per_class_acc": None, "status": "OK"}

    # ---- 1. Instantiation -------------------------------------------------
    try:
        model = build_model(name, num_classes=num_classes, config=CFG)
        _check("build_model() succeeds",    True,  model_results)
        _check("get_params() returns dict", isinstance(model.get_params(), dict), model_results)
        print(f"  Params : {model.get_params()}")
    except Exception as exc:
        _check(f"build_model() failed: {exc}", False, model_results)
        summary.update({"passed": 0, "failed": len(model_results), "status": "INIT_FAIL"})
        results.extend(model_results)
        return summary

    input_kind = "windows" if name in RAW_WINDOW_MODELS else "features"
    print(f"  Input  : {input_kind}   X_train={X_train.shape}  X_test={X_test.shape}")

    # ---- 2. fit() ---------------------------------------------------------
    try:
        t0 = time.time()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        summary["train_time"] = time.time() - t0
        _check(f"fit() completes in {summary['train_time']:.2f}s", True, model_results)
        _check("is_fitted = True", model.is_fitted, model_results)
    except Exception as exc:
        _check(f"fit() raised: {exc}", False, model_results)
        summary.update({"passed": sum(r for _, r in model_results),
                        "failed": sum(not r for _, r in model_results),
                        "status": "FIT_FAIL"})
        results.extend(model_results)
        return summary

    # ---- 3. predict() -----------------------------------------------------
    preds  = None
    n_test = len(y_test)
    try:
        preds = model.predict(X_test)
        _check("predict() returns ndarray",            isinstance(preds, np.ndarray),                      model_results)
        _check("predict() shape == (N_test,)",         preds.shape == (n_test,),                           model_results)
        _check("predict() dtype is integer",           np.issubdtype(preds.dtype, np.integer),             model_results)
        _check("predict() values in [0, num_classes)", bool(np.all((preds >= 0) & (preds < num_classes))), model_results)
        _check("predict() no NaN",                     not np.isnan(preds.astype(float)).any(),            model_results)

        test_acc = float((preds == y_test).mean())
        summary["test_acc"] = test_acc
        print(f"\n  Overall test accuracy : {test_acc:.4f}  (chance = {1/num_classes:.4f})")
    except Exception as exc:
        _check(f"predict() raised: {exc}", False, model_results)

    # ---- 4. predict_proba() -----------------------------------------------
    try:
        proba = model.predict_proba(X_test)
        _check("predict_proba() returns ndarray",            isinstance(proba, np.ndarray),                        model_results)
        _check("predict_proba() shape == (N_test, classes)", proba.shape == (n_test, num_classes),                 model_results)
        _check("predict_proba() all values in [0, 1]",       bool(np.all((proba >= 0) & (proba <= 1))),            model_results)
        _check("predict_proba() rows sum to 1 (±1e-4)",      bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)), model_results)
        _check("predict_proba() no NaN",                     not np.isnan(proba).any(),                            model_results)
        _check("predict_proba() no Inf",                     not np.isinf(proba).any(),                            model_results)
    except Exception as exc:
        _check(f"predict_proba() raised: {exc}", False, model_results)

    # ---- 5. Confusion matrix + per-class accuracy -------------------------
    if preds is not None:
        cm = _confusion_matrix(y_test, preds, num_classes)

        print(f"\n  {'─' * 50}")
        print(f"  Confusion Matrix  (rows = true, cols = predicted)")
        print(f"  {'─' * 50}")
        _print_confusion_matrix(cm, class_names)

        print(f"\n  Per-Class Accuracy")
        per_class = _print_per_class_accuracy(cm, class_names)
        summary["per_class_acc"] = per_class

    # ---- Tally ------------------------------------------------------------
    summary["passed"] = sum(1 for _, r in model_results if r)
    summary["failed"] = sum(1 for _, r in model_results if not r)
    if summary["failed"] > 0:
        summary["status"] = "PARTIAL"
    results.extend(model_results)
    return summary


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test all Emognition models against the real dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", metavar="PATH", required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--mode", default="cleaned", choices=["raw", "cleaned"],
        help="'cleaned' for *_STIMULUS_MUSE_cleaned.json, 'raw' for *_STIMULUS_MUSE.json",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL",
        help=f"Models to test. Defaults to all. Choices: {list_models()}",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs for deep models.",
    )
    args = parser.parse_args()

    # Apply epochs override
    CFG["training"]["epochs"]   = args.epochs
    CFG["training"]["patience"] = max(3, args.epochs)

    models_to_test = args.models if args.models else list_models()

    # Validate model names
    valid = set(list_models())
    bad   = [m for m in models_to_test if m not in valid]
    if bad:
        print(f"[ERROR] Unknown model(s): {bad}")
        print(f"        Valid choices   : {sorted(valid)}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Print run config
    # -----------------------------------------------------------------------
    _section("Run Configuration")
    print(f"  Dataset path   : {args.data}")
    print(f"  Dataset mode   : {args.mode}")
    print(f"  Models to test : {models_to_test}")
    print(f"  Epochs (deep)  : {args.epochs}")

    # -----------------------------------------------------------------------
    # Step 1 : Load real data
    # -----------------------------------------------------------------------
    _section("Step 1 : Loading EEG Data")
    CONFIG.MODE = args.mode
    X_raw, y, subject_ids, trial_ids, label_to_id = load_eeg_data(args.data, CONFIG)
    num_classes = len(label_to_id)
    CFG["data"]["num_classes"] = num_classes

    # Build id → label map for readable class names
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_names = [id_to_label[i] for i in range(num_classes)]

    print(f"  X_raw shape  : {X_raw.shape}")
    print(f"  Classes      : {label_to_id}")
    print(f"  Subjects     : {sorted(set(subject_ids.tolist()))}")

    # -----------------------------------------------------------------------
    # Step 2 : Create splits
    # -----------------------------------------------------------------------
    _section("Step 2 : Creating Data Splits")
    splits = create_data_splits(y, subject_ids, CONFIG, trial_ids=trial_ids)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"  Train : {len(tr)}  Val : {len(va)}  Test : {len(te)}")

    # -----------------------------------------------------------------------
    # Step 3 : Extract statistical features (classical ML + MLP)
    # -----------------------------------------------------------------------
    _section("Step 3 : Extracting Statistical Features")
    CONFIG.FEATURE_MODE = "statistical"
    X_feat = extract_eeg_features(X_raw, CONFIG)
    print(f"  Feature matrix : {X_feat.shape}")

    X_feat_train, X_feat_val, X_feat_test = X_feat[tr], X_feat[va], X_feat[te]
    X_raw_train,  X_raw_val,  X_raw_test  = X_raw[tr],  X_raw[va],  X_raw[te]
    y_train,      y_val,      y_test       = y[tr],      y[va],      y[te]

    print(f"  Feature splits : train={X_feat_train.shape}  val={X_feat_val.shape}  test={X_feat_test.shape}")
    print(f"  Window  splits : train={X_raw_train.shape}   val={X_raw_val.shape}   test={X_raw_test.shape}")

    # -----------------------------------------------------------------------
    # Run tests
    # -----------------------------------------------------------------------
    all_results:   list = []
    all_summaries: list = []

    for name in models_to_test:
        Xtr, Xva, Xte = (
            (X_raw_train,  X_raw_val,  X_raw_test)
            if name in RAW_WINDOW_MODELS else
            (X_feat_train, X_feat_val, X_feat_test)
        )
        summary = test_one_model(
            name,
            Xtr, y_train,
            Xva, y_val,
            Xte, y_test,
            num_classes,
            class_names,
            all_results,
        )
        all_summaries.append(summary)

    # -----------------------------------------------------------------------
    # Final summary table — overall accuracy
    # -----------------------------------------------------------------------
    _section("FINAL SUMMARY — Overall Accuracy")

    col = "{:<14}  {:>6}  {:>6}  {:>10}  {:>10}  {}"
    print("  " + col.format("Model", "Passed", "Failed", "Train(s)", "Test Acc", "Status"))
    print("  " + "-" * 66)

    for s in all_summaries:
        acc = f"{s['test_acc']:.4f}" if s["test_acc"] is not None else "   n/a  "
        print("  " + col.format(
            s["name"],
            s["passed"],
            s["failed"],
            f"{s['train_time']:.2f}",
            acc,
            s["status"],
        ))

    # -----------------------------------------------------------------------
    # Final summary table — per-class accuracy
    # -----------------------------------------------------------------------
    _section("FINAL SUMMARY — Per-Class Accuracy")

    # Header
    hdr = f"  {'Model':<14}"
    for c in class_names:
        hdr += f"  {c:>12}"
    hdr += f"  {'Overall':>12}"
    print(hdr)
    print("  " + "-" * (14 + (len(class_names) + 1) * 14))

    for s in all_summaries:
        row = f"  {s['name']:<14}"
        if s["per_class_acc"] is not None:
            for acc in s["per_class_acc"]:
                row += f"  {acc:>11.4f} "
            row += f"  {s['test_acc']:>11.4f} "
        else:
            row += "  " + "   n/a       " * (len(class_names) + 1)
        print(row)

    # -----------------------------------------------------------------------
    # Totals
    # -----------------------------------------------------------------------
    total_passed = sum(s["passed"] for s in all_summaries)
    total_failed = sum(s["failed"] for s in all_summaries)
    total_checks = total_passed + total_failed

    print(f"\n  Dataset : {args.data}  ({args.mode})")
    print(f"  Epochs  : {args.epochs}")
    print(f"\n  {total_passed} passed  |  {total_failed} failed  |  {total_checks} total checks")
    print("=" * 70 + "\n")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
