import os

# Base directories
ARTIFACTS_DIR = "artifacts"
RAW_DIRS = os.path.join(ARTIFACTS_DIR, "raw")
PREPROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "preprocessed")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")

# Config file
CONFIG_PATH = os.path.join("config", "config.yaml")

# File names inside artifacts
RAW_DATA_FILE = os.path.join(RAW_DIRS, "enhanced_credit_card_fraud_dataset.csv")
PREPROCESSED_FILE = os.path.join(PREPROCESSED_DIR, "preprocessed.pkl")
MODEL_FILE = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
TUNED_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")
EVALUATION_FILE = os.path.join(REPORTS_DIR, "evaluation_results.pkl")

# Ensure directories exist
os.makedirs(RAW_DIRS, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
