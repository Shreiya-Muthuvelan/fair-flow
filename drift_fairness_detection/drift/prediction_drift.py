# drift_fairness_detection/drift/prediction_drift.py
import numpy as np
from scipy import stats
from drift_fairness_detection.config import (APPROVAL_RATE_DELTA_THRESHOLD,
                            CONFIDENCE_MEAN_DELTA_THRESHOLD)

def monitor_prediction_drift(y_prob: np.ndarray,
                              y_pred: np.ndarray,
                              baseline_stats: dict) -> dict:

    baseline_approval = baseline_stats["target"]["approval_rate"]
    current_approval  = float(y_pred.mean())
    approval_delta    = abs(current_approval - baseline_approval)

    # KS test on probability scores vs baseline confidence values
    baseline_probs = baseline_stats.get("confidence", {}).get("values", [])
    if baseline_probs:
        ks_stat, ks_p = stats.ks_2samp(baseline_probs, y_prob.tolist())
        conf_drift = ks_p < 0.05
    else:
        ks_stat, ks_p, conf_drift = 0.0, 1.0, False

    return {
        "approval_rate": {
            "current": round(current_approval, 4),
            "baseline": round(baseline_approval, 4),
            "delta": round(approval_delta, 4),
            "drift_detected": approval_delta > APPROVAL_RATE_DELTA_THRESHOLD
        },
        "confidence_distribution": {
            "current_mean": round(float(y_prob.mean()), 4),
            "current_std":  round(float(y_prob.std()),  4),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value":   round(ks_p, 4),
            "drift_detected": conf_drift
        }
    }