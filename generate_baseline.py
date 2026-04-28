import pandas as pd
import json
from pathlib import Path
from preprocessing.config import SPLITS_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from model.model_config import DECISION_THRESHOLD

X_train = pd.read_parquet(f"{SPLITS_DIR}X_train.parquet")
y_train = pd.read_parquet(f"{SPLITS_DIR}y_train.parquet").squeeze()

features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

baseline = {
    "n_samples": len(X_train),
    "decision_threshold": DECISION_THRESHOLD,
    "numerical": {},
    "categorical": {},
    "target": {
        "approval_rate": float((y_train == 1).mean()),
        "class_counts": y_train.value_counts().to_dict()
    }
}

for col in NUMERICAL_FEATURES:
    if col not in X_train.columns:
        continue
    s = X_train[col]
    baseline["numerical"][col] = {
        "mean": float(s.mean()),
        "std":  float(s.std()),
        "min":  float(s.min()),
        "max":  float(s.max()),
        "p25":  float(s.quantile(0.25)),
        "p75":  float(s.quantile(0.75)),
        "values": s.tolist()   # kept for KS test in Layer 4
    }

for col in CATEGORICAL_FEATURES:
    if col not in X_train.columns:
        continue
    baseline["categorical"][col] = X_train[col].value_counts().to_dict()

Path("artifacts").mkdir(exist_ok=True)
with open("artifacts/baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print("Baseline saved to artifacts/baseline.json")
print(f"  Numerical features: {list(baseline['numerical'].keys())}")
print(f"  Categorical features: {list(baseline['categorical'].keys())}")

