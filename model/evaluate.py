import pandas as pd, pickle
from sklearn.metrics import classification_report, roc_auc_score
from preprocessing.config import SPLITS_DIR, MODEL_PATH, SENSITIVE_ATTRS
from .model_config import MODEL_FEATURES, DECISION_THRESHOLD

def evaluate_model(split="val"):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X = pd.read_parquet(f"{SPLITS_DIR}X_{split}.parquet")
    y = pd.read_parquet(f"{SPLITS_DIR}y_{split}.parquet").squeeze()

    features = [c for c in MODEL_FEATURES if c in X.columns]
    y_prob = model.predict_proba(X[features])[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)  # always use locked threshold

    print(f"\Evaluation on val set (threshold={DECISION_THRESHOLD}) ")
    print(classification_report(y, y_pred, target_names=["bad", "good"]))
    print(f"ROC-AUC: {roc_auc_score(y, y_prob):.4f}")
    return y_pred, y_prob

if __name__ == "__main__":
    evaluate_model()