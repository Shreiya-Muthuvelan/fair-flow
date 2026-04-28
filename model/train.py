import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from .model_config import XGBOOST_PARAMS, MODEL_FEATURES
from preprocessing.config import SPLITS_DIR, MODEL_PATH, SENSITIVE_ATTRS

def train_model():
    X_train = pd.read_parquet(f"{SPLITS_DIR}X_train.parquet")
    y_train = pd.read_parquet(f"{SPLITS_DIR}y_train.parquet").squeeze()

    features = ["Age", "Credit amount", "Duration", "Debt_ratio",
                "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]

    print("Feature dtypes going into model:")
    print(X_train[features].dtypes)
    print("\nAny nulls:", X_train[features].isnull().sum().sum())
    print("y_train unique:", y_train.unique(), "dtype:", y_train.dtype)

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train[features], y_train)  # bare minimum fit

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Model saved")
    return model


def find_best_threshold(model, X_val, y_val, features):
    """
    Scan thresholds from 0.2 to 0.6 and pick the one that
    maximises F1 on the 'bad' class (class 0).
    This is the optimisation target for credit risk.
    """
    X_val_model = X_val[features]
    y_prob = model.predict_proba(X_val_model)[:, 1]

    best_thresh = 0.5
    best_f1_bad = 0.0

    print("\nThreshold scan (optimising for 'bad' class F1):")
    print(f"{'Threshold':>10}  {'Bad-Precision':>14}  {'Bad-Recall':>10}  {'Bad-F1':>8}  {'ROC-AUC':>8}")

    for thresh in np.arange(0.2, 0.65, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, pos_label=0, average="binary", zero_division=0
        )
        auc = roc_auc_score(y_val, y_prob)
        print(f"{thresh:>10.2f}  {prec:>14.3f}  {rec:>10.3f}  {f1:>8.3f}  {auc:>8.4f}")

        if f1 > best_f1_bad:
            best_f1_bad = f1
            best_thresh = thresh

    print(f"\nBest threshold: {best_thresh:.2f}  (bad-class F1 = {best_f1_bad:.3f})")
    return best_thresh


def evaluate_model(model=None, split="val", threshold=None):
    if model is None:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    X = pd.read_parquet(f"{SPLITS_DIR}X_{split}.parquet")
    y = pd.read_parquet(f"{SPLITS_DIR}y_{split}.parquet").squeeze()

    features = [c for c in MODEL_FEATURES if c in X.columns]
    y_prob = model.predict_proba(X[features])[:, 1]

    if threshold is None:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nEvaluation on {split} set (threshold={threshold:.2f})")
    print(classification_report(y, y_pred, target_names=["bad", "good"]))
    print(f"ROC-AUC: {roc_auc_score(y, y_prob):.4f}")
    return y_pred, y_prob