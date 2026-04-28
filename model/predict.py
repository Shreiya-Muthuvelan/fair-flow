import pickle
import pandas as pd
from .model_config import MODEL_FEATURES, DECISION_THRESHOLD
from preprocessing.config import MODEL_PATH

_model_cache = None  

def load_model():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
    return _model_cache

def predict_batch(X_batch: pd.DataFrame):
    model = load_model()
    features = [c for c in MODEL_FEATURES if c in X_batch.columns]
    y_prob = model.predict_proba(X_batch[features])[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    return y_pred, y_prob