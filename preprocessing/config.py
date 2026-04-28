# preprocessing/config.py

RAW_DATA_PATH = "data/raw/german_credit_data.csv"

TARGET_COL = "Risk"


DRIFT_FEATURES = ["Age", "Job", "Credit amount", "Duration", "Debt_ratio"]


SENSITIVE_ATTRS = ["Sex", "Age_group"]  

NUMERICAL_FEATURES = ["Age", "Credit amount", "Duration", "Debt_ratio"]

CATEGORICAL_FEATURES = ["Job", "Housing", "Saving accounts", 
                        "Checking account", "Purpose"]
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

PASSTHROUGH_COLS = SENSITIVE_ATTRS


SPLIT_SIZES = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_STATE = 42

# Artifact output paths
ARTIFACTS_DIR = "artifacts/"
SPLITS_DIR    = "artifacts/splits/"
MODEL_PATH    = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"
BASELINE_PATH = "artifacts/baseline.json"