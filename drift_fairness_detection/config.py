NUMERICAL_DRIFT_FEATURES   = ["Age", "Credit amount", "Duration", "Debt_ratio"]
CATEGORICAL_DRIFT_FEATURES = ["Job", "Housing", "Saving accounts",
                               "Checking account", "Purpose"]
FAIRNESS_ATTRS   = ["Sex", "Age_group"]
MODEL_FEATURES   = ["Age", "Credit amount", "Duration", "Debt_ratio",
                    "Job", "Housing", "Saving accounts",
                    "Checking account", "Purpose"]
DECISION_THRESHOLD = 0.35
MIN_GROUP_SIZE = 20  # minimum samples per group to compute fairness metrics
# Drift thresholds
KS_P_VALUE_THRESHOLD  = 0.05
KL_WARNING_THRESHOLD  = 0.1
KL_CRITICAL_THRESHOLD = 0.3

# Fairness thresholds
DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
EQUAL_OPP_THRESHOLD          = 0.1
DISPARATE_IMPACT_THRESHOLD   = 0.8

# Prediction drift
APPROVAL_RATE_DELTA_THRESHOLD    = 0.15
CONFIDENCE_MEAN_DELTA_THRESHOLD  = 0.1

# Paths
BASELINE_PATH = "artifacts/baseline.json"
MODEL_PATH    = "artifacts/model.pkl"
SPLITS_DIR    = "artifacts/splits/"
DB_PATH       = "artifacts/fairflow.db"