import pandas as pd
import numpy as np
from scipy.stats import chisquare
from drift_fairness_detection.config import KS_P_VALUE_THRESHOLD

def chi_square_test(baseline_freq: dict, current_series: pd.Series, feature: str) -> dict:
    # Align categories
    all_cats = sorted(set(baseline_freq.keys()) |
                      set(current_series.astype(str).unique()))

    observed = []
    expected = []
    n_current  = len(current_series)
    n_baseline = sum(baseline_freq.values())

    for cat in all_cats:
        obs = (current_series.astype(str) == str(cat)).sum()
        exp = baseline_freq.get(str(cat), 0) / n_baseline * n_current
        observed.append(obs)
        expected.append(max(exp, 1e-9))   # avoid zero division

    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)

    # Scale expected to match observed total
    expected = expected / expected.sum() * observed.sum()

    stat, p_value = chisquare(observed, f_exp=expected)

    if p_value > KS_P_VALUE_THRESHOLD:
        severity = "INFO"
    elif stat > 10:
        severity = "CRITICAL"
    else:
        severity = "WARNING"

    return {
        "feature": feature,
        "chi2_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": p_value < KS_P_VALUE_THRESHOLD,
        "severity": severity,
        "category_counts": {
            cat: int((current_series.astype(str) == str(cat)).sum())
            for cat in all_cats
        }
    }


def run_chi_square_drift(baseline_stats: dict, current_df, features: list) -> dict:
    results = {}
    for feature in features:
        if feature not in current_df.columns:
            continue
        if feature not in baseline_stats.get("categorical", {}):
            continue
        baseline_freq = baseline_stats["categorical"][feature]
        results[feature] = chi_square_test(
            baseline_freq, current_df[feature], feature
        )
    return results