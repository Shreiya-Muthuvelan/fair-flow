import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report

SPLITS_DIR   = "artifacts/splits/"
MODEL_PATH   = "artifacts/model.pkl"
BASELINE_DIR = "artifacts/baseline/"
Path(BASELINE_DIR).mkdir(parents=True, exist_ok=True)

MODEL_FEATURES = ["Age", "Credit amount", "Duration", "Debt_ratio",
                  "Job", "Housing", "Saving accounts",
                  "Checking account", "Purpose"]
SENSITIVE_COLS = ["Sex", "Age_group"]   
DECISION_THRESHOLD = 0.35

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

X_val   = pd.read_parquet(f"{SPLITS_DIR}X_val.parquet")
y_val   = pd.read_parquet(f"{SPLITS_DIR}y_val.parquet").squeeze()
X_train = pd.read_parquet(f"{SPLITS_DIR}X_train.parquet")
X_test  = pd.read_parquet(f"{SPLITS_DIR}X_test.parquet")

with open("artifacts/feature_names.json", "w") as f:
    json.dump(MODEL_FEATURES, f, indent=2)
print(" feature_names.json")

for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
    sens_cols = [c for c in SENSITIVE_COLS if c in X.columns]
    X[sens_cols].to_parquet(f"{SPLITS_DIR}sensitive_{split_name}.parquet", index=False)
print(" sensitive_train/val/test.parquet")

# Initial Fairness Audit
y_prob = model.predict_proba(X_val[MODEL_FEATURES])[:, 1]
y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

audit = {"generated_at": datetime.now().isoformat(), "split": "val",
         "threshold": DECISION_THRESHOLD, "by_sex": {}, "by_age_group": {}}

pred_series = pd.Series(y_pred, index=X_val.index)

# demographic parity + disparate impact by Sex
if "Sex" in X_val.columns:
    sex_rates = {}
    for grp, idx in X_val.groupby("Sex").groups.items():
        rate = float(pred_series.loc[idx].mean())
        sex_rates[str(grp)] = rate
    rates = list(sex_rates.values())
    audit["by_sex"] = {
        "approval_rates": sex_rates,
        "demographic_parity_difference": round(max(rates) - min(rates), 4),
        "disparate_impact_ratio": round(min(rates) / max(rates), 4),
        "demographic_parity_violation": (max(rates) - min(rates)) > 0.1,
        "disparate_impact_violation": (min(rates) / max(rates)) < 0.8,
    }

# equal opportunity by Sex (TPR among y_true==1)
if "Sex" in X_val.columns:
    tpr = {}
    for grp, idx in X_val.groupby("Sex").groups.items():
        mask = (y_val.loc[idx] == 1)
        tpr[str(grp)] = float(pred_series.loc[idx][mask].mean()) if mask.sum() > 0 else None
    tpr_vals = [v for v in tpr.values() if v is not None]
    audit["by_sex"]["equal_opportunity_difference"] = round(max(tpr_vals) - min(tpr_vals), 4)
    audit["by_sex"]["equal_opportunity_violation"] = (max(tpr_vals) - min(tpr_vals)) > 0.1

# repeat for Age_group
if "Age_group" in X_val.columns:
    age_rates = {}
    for grp, idx in X_val.groupby("Age_group").groups.items():
        age_rates[str(grp)] = float(pred_series.loc[idx].mean())
    rates = list(age_rates.values())
    audit["by_age_group"] = {
        "approval_rates": age_rates,
        "demographic_parity_difference": round(max(rates) - min(rates), 4),
        "disparate_impact_ratio": round(min(rates) / max(rates), 4),
        "demographic_parity_violation": (max(rates) - min(rates)) > 0.1,
        "disparate_impact_violation": (min(rates) / max(rates)) < 0.8,
    }

with open("artifacts/baseline/fairness_audit_report.json", "w") as f:
    json.dump(audit, f, indent=2)
print("fairness_audit_report.json")

report = classification_report(y_val, y_pred, output_dict=True)
metadata = {
    "model_version": "1.0",
    "model_file": "artifacts/model.pkl",
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "decision_threshold": DECISION_THRESHOLD,
    "hyperparameters": {
        "n_estimators": model.n_estimators,
        "max_depth": int(model.max_depth),
        "learning_rate": float(model.learning_rate),
        "scale_pos_weight": float(model.scale_pos_weight),
    },
    "performance_metrics": {
        "auc_roc": round(roc_auc_score(y_val, y_prob), 4),
        "accuracy": round(report["accuracy"], 4),
        "bad_precision": round(report["0"]["precision"], 4),
        "bad_recall": round(report["0"]["recall"], 4),
        "good_recall": round(report["1"]["recall"], 4),
        "demographic_parity_by_sex": audit.get("by_sex", {}).get("demographic_parity_difference"),
        "disparate_impact_by_sex": audit.get("by_sex", {}).get("disparate_impact_ratio"),
    },
    "training_samples": 700,
    "validation_samples": 150,
    "test_samples": 150,
    "features": MODEL_FEATURES,
    "sensitive_attributes": SENSITIVE_COLS,
}

with open("artifacts/baseline/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("metadata.json")
print("\nFairness Audit Summary")
print(json.dumps(audit, indent=2))