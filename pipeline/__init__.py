"""
pipeline — Emognition core modules
"""
from pipeline.data_loaders.data_loader          import load_eeg_data, create_data_splits
from pipeline.feature_extraction.feature_extractor import extract_eeg_features, get_feature_names, clear_model_cache
from pipeline.preprocessing.baseline_reduction  import apply_baseline_reduction, load_baseline_files
