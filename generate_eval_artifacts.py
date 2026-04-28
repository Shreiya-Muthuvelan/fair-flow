import pandas as pd
import pickle
import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance

MODEL_FEATURES = ["Age", "Credit amount", "Duration", "Debt_ratio",
                  "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
THRESHOLD = 0.35

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

X_val = pd.read_parquet("artifacts/splits/X_val.parquet")
y_val = pd.read_parquet("artifacts/splits/y_val.parquet").squeeze()

y_prob = model.predict_proba(X_val[MODEL_FEATURES])[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# 1. confusion_matrix.csv
cm = confusion_matrix(y_val, y_pred)
pd.DataFrame(cm,
    index=["Actual Bad", "Actual Good"],
    columns=["Predicted Bad", "Predicted Good"]
).to_csv("artifacts/baseline/confusion_matrix.csv")

# 2. roc_curve_data.csv
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}
).to_csv("artifacts/baseline/roc_curve_data.csv", index=False)

# 3. feature_importance.csv
importance = model.feature_importances_
pd.DataFrame({
    "feature": MODEL_FEATURES,
    "importance": importance
}).sort_values("importance", ascending=False
).to_csv("artifacts/baseline/feature_importance.csv", index=False)

print("✓ confusion_matrix.csv")
print("✓ roc_curve_data.csv")
print("✓ feature_importance.csv")