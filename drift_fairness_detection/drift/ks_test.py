from scipy import stats
from drift_fairness_detection.config import KS_P_VALUE_THRESHOLD

def ks_test(reference_values: list, current_values: list, feature: str) -> dict:
    stat, p_value = stats.ks_2samp(reference_values, current_values)

    if p_value > KS_P_VALUE_THRESHOLD:
        severity = "INFO"
    elif stat > 0.3:
        severity = "CRITICAL"
    else:
        severity = "WARNING"

    return {
        "feature": feature,
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": p_value < KS_P_VALUE_THRESHOLD,
        "severity": severity
    }


def run_ks_drift(reference_df, current_df, features: list) -> dict:
    results = {}
    for feature in features:
        if feature not in current_df.columns:
            continue
        ref_vals = reference_df[feature].dropna().tolist()
        cur_vals = current_df[feature].dropna().tolist()
        results[feature] = ks_test(ref_vals, cur_vals, feature)
    return results