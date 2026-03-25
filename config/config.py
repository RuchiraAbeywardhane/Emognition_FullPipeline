"""
config.py — Master Pipeline Configuration
============================================================
Single source of truth for the entire Emognition emotion-recognition
pipeline.  Every module (data_loader, feature_extractor, models,
trainer, evaluator) reads from here.

How to use
----------
  from config import CFG

  # Access any value
  CFG["data"]["fs"]
  CFG["features"]["active"]
  CFG["models"]["active"]

  # Override a value for a quick experiment (does NOT mutate the file)
  import copy
  cfg = copy.deepcopy(CFG)
  cfg["training"]["epochs"] = 5

Sections
--------
  data        : Dataset paths, loader mode, windowing, quality filter
  baseline    : Baseline reduction method and parameters
  features    : Feature extractors to run and their parameters
  models      : Models to train and their hyperparameters
  training    : Optimiser, scheduler, epochs, batch size, splits
  evaluation  : Metrics, cross-validation, output paths
  logging     : Verbosity, result saving
  reproducibility : Seeds

Author : Final Year Project
Date   : 2026
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Convenience alias — import this in every other module
#   from config import CFG
# ---------------------------------------------------------------------------

CFG: dict = {

    # =======================================================================
    # DATA
    # =======================================================================
    "data": {
        # ---- Paths --------------------------------------------------------
        # Root directory that contains raw or cleaned recordings.
        # Override this to point at your local copy of the dataset.
        "raw_root":     r"data/dataset1_emognition/raw",
        "cleaned_root": r"data/dataset1_emognition/processed",

        # Which root is fed to load_eeg_data().
        # 'raw'     → JSON MUSE files, windowing + feature extraction applied here
        # 'cleaned' → pre-processed CSVs / JSONs
        "mode": "raw",

        # ---- Signal -------------------------------------------------------
        # MUSE headband sample rate (Hz)
        "fs": 256.0,

        # EEG channel names expected in the files
        "eeg_cols": ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"],

        # ---- Windowing ----------------------------------------------------
        "win_sec": 4.0,        # window length in seconds
        "overlap": 0.5,        # fractional overlap  [0, 1)

        # ---- Quality filter (raw mode only) ------------------------------
        "quality_filter": True,
        # MUSE HSI scale: 1 = good, 2 = ok, 4 = bad
        # Samples with any HSI value ABOVE this threshold are dropped
        "hsi_threshold": 2,

        # ---- Labels -------------------------------------------------------
        # Keyword in filenames that marks a baseline (rest) recording
        "baseline_keyword": "baseline",

        # Map the raw string labels in the dataset to integer class indices.
        # Add / remove entries to match your annotation scheme.
        "label_map": {
            "enthusiasm": 0,
            "sadness":    1,
            "fear":       2,
            "neutral":    3,
        },

        # Number of emotion classes (must match len(label_map))
        "num_classes": 4,
    },

    # =======================================================================
    # BASELINE REDUCTION
    # =======================================================================
    "baseline": {
        # Method applied by baseline_reduction.py
        # Options: 'invbase' | 'zscore' | 'subtract'
        "method": "invbase",

        # Small constant to avoid division by zero
        "eps": 1e-12,

        # If True, reduce_dataset() skips files that already exist
        "overwrite": False,
    },

    # =======================================================================
    # FEATURE EXTRACTION
    # =======================================================================
    "features": {
        # Which extractors are active for this run.
        # Each name maps to a key in the 'extractors' dict below.
        # You can list multiple — the pipeline will run all of them
        # and produce one feature matrix per extractor.
        "active": ["statistical"],   # e.g. ["statistical", "welch_psd", "hjorth"]

        "extractors": {

            # ---- Statistical (current default — 26 features / channel) ---
            "statistical": {
                "bands": {
                    "delta": [0.5,  4.0],
                    "theta": [4.0,  8.0],
                    "alpha": [8.0,  13.0],
                    "beta":  [13.0, 30.0],
                    "gamma": [30.0, 45.0],
                },
                # Features computed per band: 'psd', 'de' (differential entropy)
                "band_features": ["psd", "de"],
                # Temporal features computed per channel
                "temporal_features": [
                    "mean", "std", "var", "rms", "skew", "kurtosis",
                    "p2p", "zcr", "energy", "mobility", "complexity",
                    "iqr", "max", "min", "median", "snr",
                ],
                # --- NEW: extra per-channel feature groups ---
                # Wavelet DWT subband energies (requires PyWavelets)
                "wavelet": {
                    "enabled": True,
                    "wavelet": "db4",
                    "level":   5,        # number of decomposition levels
                },
                # Entropy features
                "entropy": {
                    "spectral_entropy":    True,   # Welch-based normalised SE
                    "permutation_entropy": True,   # PE order=3, delay=1
                },
                # Per-channel alpha/beta, alpha/theta, theta/beta ratios
                "band_ratios": True,
                # --- NEW: cross-channel features ---
                # Mean coherence per band for all 6 channel pairs → 30 features
                "coherence": True,
                # Frontal (AF7-AF8) and lateral (TP9-TP10) band-power asymmetry → 10
                "asymmetry": True,
                "eps": 1e-12,
            },

            # ---- Welch PSD (placeholder — add extractor module later) ----
            "welch_psd": {
                "nperseg": 256,        # FFT segment length (samples)
                "noverlap": 128,
                "bands": {
                    "delta": [0.5,  4.0],
                    "theta": [4.0,  8.0],
                    "alpha": [8.0,  13.0],
                    "beta":  [13.0, 30.0],
                    "gamma": [30.0, 45.0],
                },
            },

            # ---- Hjorth parameters (placeholder) -------------------------
            "hjorth": {
                # activity, mobility, complexity per channel = 3 × n_channels
            },

            # ---- Wavelet (placeholder) ------------------------------------
            "wavelet": {
                "wavelet":   "db4",
                "level":     4,
                "features":  ["mean", "std", "energy"],
            },
        },
    },

    # =======================================================================
    # MODELS
    # =======================================================================
    "models": {
        # Which models are active for this run.
        # Each name maps to a key in the 'architectures' dict below.
        "active": ["svm"],   # e.g. ["svm", "random_forest", "mlp", "lstm", "cnn1d"]

        "architectures": {

            # ---- Classical ML --------------------------------------------
            "svm": {
                "kernel":  "rbf",
                "C":       1.0,
                "gamma":   "scale",
                "probability": True,   # needed for predict_proba
            },

            "random_forest": {
                "n_estimators": 200,
                "max_depth":    None,
                "n_jobs":       -1,
            },

            "knn": {
                "n_neighbors": 5,
                "metric":      "euclidean",
            },

            # --- UPDATED LDA config ---
            "lda": {
                # 'lsqr' supports shrinkage; 'svd' does not
                "solver":         "lsqr",
                # Regularisation: 'auto' (Ledoit-Wolf), float 0-1, or None
                "shrinkage":      "auto",
                # Top-K features selected by mutual information before LDA.
                # Set to None to keep all features.
                "n_features_mi":  120,
                # Grid-search shrinkage on the validation split when True
                "tune_shrinkage": True,
                "shrinkage_grid": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, "auto"],
            },

            # ---- Deep Learning -------------------------------------------
            "mlp": {
                # Layer sizes EXCLUDING input (determined by feature dim)
                "hidden_layers": [256, 128, 64],
                "dropout":       0.3,
                "activation":    "relu",
            },

            "lstm": {
                # Input is (batch, seq_len, n_channels) — raw windows
                "hidden_size":  128,
                "num_layers":   2,
                "dropout":      0.3,
                "bidirectional": False,
            },

            "bilstm": {
                "hidden_size":  128,
                "num_layers":   2,
                "dropout":      0.3,
                "bidirectional": True,
            },

            "cnn1d": {
                # Operates on (batch, n_channels, seq_len) — raw windows
                "filters":      [32, 64, 128],
                "kernel_size":  3,
                "pool_size":    2,
                "dropout":      0.3,
            },

            "cnn_lstm": {
                # CNN feature extractor followed by LSTM
                "cnn_filters":      [32, 64],
                "cnn_kernel_size":  3,
                "lstm_hidden":      128,
                "dropout":          0.3,
            },

            "transformer": {
                "d_model":    64,
                "nhead":      4,
                "num_layers": 2,
                "dropout":    0.1,
                "dim_feedforward": 256,
            },
        },
    },

    # =======================================================================
    # TRAINING
    # =======================================================================
    "training": {
        # ---- Split strategy -----------------------------------------------
        # 'random'               — shuffled, no subject/trial awareness
        # 'subject_independent'  — whole subjects kept in one split only
        # 'clip_independent'     — whole trials kept in one split only
        "split_strategy": "subject_independent",
        "train_ratio": 0.70,
        "val_ratio":   0.15,
        "test_ratio":  0.15,

        # ---- Cross-validation (set n_folds > 1 to enable) ----------------
        "cross_validation": False,
        "n_folds":          5,       # used when cross_validation = True

        # ---- Optimiser (deep learning models) ----------------------------
        "optimizer":    "adam",      # 'adam' | 'sgd' | 'adamw'
        "learning_rate": 1e-3,
        "weight_decay":  1e-4,

        # ---- Scheduler ---------------------------------------------------
        "scheduler":       "plateau",   # 'plateau' | 'cosine' | 'step' | None
        "scheduler_params": {
            "patience": 5,             # for ReduceLROnPlateau
            "factor":   0.5,
            "min_lr":   1e-6,
        },

        # ---- Loop --------------------------------------------------------
        "epochs":           50,
        "batch_size":       32,
        "early_stopping":   True,
        "patience":         10,        # epochs without val-loss improvement

        # ---- Class imbalance ---------------------------------------------
        # 'none' | 'class_weight' | 'oversample' | 'smote'
        "imbalance_strategy": "class_weight",

        # ---- Per-subject normalisation ------------------------------------
        # When True, each subject's windows are z-scored independently
        # before being passed to classical ML models. This matches the
        # normalisation strategy used in the reference pipeline and reduces
        # inter-subject amplitude variability.
        "per_subject_norm": True,
    },

    # =======================================================================
    # EVALUATION
    # =======================================================================
    "evaluation": {
        # Metrics computed on the test set for every model
        "metrics": [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "cohen_kappa",
            "confusion_matrix",
            "roc_auc",          # requires probability outputs
        ],

        # Save per-fold results when cross_validation = True
        "save_fold_results": True,

        # Where to write result CSVs, plots, and saved models
        "output_dir": r"results/dataset1",
    },

    # =======================================================================
    # LOGGING
    # =======================================================================
    "logging": {
        # 0 = silent, 1 = progress, 2 = verbose
        "verbosity": 1,

        # Save training loss/accuracy curves to output_dir
        "save_curves": True,

        # Print a comparison table of all active models at the end
        "compare_models": True,
    },

    # =======================================================================
    # REPRODUCIBILITY
    # =======================================================================
    "reproducibility": {
        "seed": 42,
    },
}


# ---------------------------------------------------------------------------
# Flat convenience helpers used by tests and simple scripts
# ---------------------------------------------------------------------------

# The dict passed directly to load_eeg_data() and create_data_splits()
LOADER_CFG: dict = {
    "mode":             CFG["data"]["mode"],
    "fs":               CFG["data"]["fs"],
    "win_sec":          CFG["data"]["win_sec"],
    "overlap":          CFG["data"]["overlap"],
    "quality_filter":   CFG["data"]["quality_filter"],
    "hsi_threshold":    CFG["data"]["hsi_threshold"],
    "baseline_keyword": CFG["data"]["baseline_keyword"],
    "label_map":        CFG["data"]["label_map"],
    "split_strategy":   CFG["training"]["split_strategy"],
    "seed":             CFG["reproducibility"]["seed"],
}

# ---------------------------------------------------------------------------
# Simple config object — used by data_loader.py and test_pipeline.py
# ---------------------------------------------------------------------------

class _Config:
    """Attribute-style config wrapper consumed by data_loader.py."""

    # Sampling & windowing
    EEG_FS          = 256.0   # MUSE sampling rate (Hz)
    EEG_WINDOW_SEC  = 4.0     # Window length (seconds)
    EEG_OVERLAP     = 0.5     # Overlap fraction  [0, 1)

    # Mode: 'raw' for *_STIMULUS_MUSE.json, 'cleaned' for *_STIMULUS_MUSE_cleaned.json
    MODE = "cleaned"

    # Baseline reduction
    USE_BASELINE_REDUCTION = True   # Set True to enable InvBase

    # Split strategy — pick ONE
    SUBJECT_INDEPENDENT = True
    CLIP_INDEPENDENT    = False

    # Reproducibility
    SEED = 42

    # Emotion → superclass mapping
    # Keys must match the second "_"-delimited part of each filename (upper-cased)
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "enthusiasm",
        "SADNESS":   "sadness",
        "FEAR":      "fear",
        "NEUTRAL":   "neutral",
    }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    # 'statistical' — hand-crafted features only         → (N, C×26)
    # 'deep'        — learned embedding only             → (N, DEEP_EMBED_DIM)
    # 'combined'    — statistical + deep concatenated    → (N, C×26 + DEEP_EMBED_DIM)
    FEATURE_MODE    = "statistical"

    # Deep extractor model: 'cnn1d' | 'lstm' | 'cnn_lstm' | 'transformer'
    DEEP_EXTRACTOR  = "cnn1d"

    # Output embedding size for deep extractors
    DEEP_EMBED_DIM  = 128

    # Batch size used when running deep extractors
    DEEP_BATCH_SIZE = 256

    # Device for PyTorch inference: 'cpu' or 'cuda'
    DEVICE          = "cpu"

    # Numerical stability constant
    EPS             = 1e-12

    # Per-subject normalisation (used by data_loader / trainer)
    PER_SUBJECT_NORM = True


CONFIG = _Config()
