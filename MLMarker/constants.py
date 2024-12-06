from pathlib import Path

# Base directory dynamically determined relative to the package
BASE_DIR = Path(__file__).resolve().parent / "models"

# Binary model paths
BINARY_MODEL_PATH = BASE_DIR / "binary_TP_4000features_95to75missingness_2024.joblib"
BINARY_FEATURES_PATH = BASE_DIR / "binary_features_TP_4000features_95to75missingness_2024.txt"

# Multi-class model paths
MULTI_CLASS_MODEL_PATH = BASE_DIR / "TP_full_92%_10exp_2024.joblib"
MULTI_CLASS_FEATURES_PATH = BASE_DIR / "features_TP_full_92%_10exp_2024.txt"
