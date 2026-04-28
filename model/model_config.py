
from xgboost.callback import EarlyStopping

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 2.33,
    "min_child_weight": 3,
    "gamma": 0.1,
    "random_state": 42,
    "use_label_encoder": False,
}
DECISION_THRESHOLD = 0.35
from preprocessing.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES