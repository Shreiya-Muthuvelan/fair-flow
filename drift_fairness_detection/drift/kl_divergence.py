# drift_fairness_detection/drift/kl_divergence.py
import numpy as np
from scipy.special import rel_entr
from drift_fairness_detection.config import KL_WARNING_THRESHOLD, KL_CRITICAL_THRESHOLD

def _to_distribution(values, bins, bin_edges=None):
    if bin_edges is None:
        counts, bin_edges = np.histogram(values, bins=bins)
    else:
        counts, _ = np.histogram(values, bins=bin_edges)
    counts = counts.astype(float) + 1e-9
    return counts / counts.sum(), bin_edges


def kl_divergence_numerical(ref_values, cur_values, feature, bins=20) -> dict:
    p, bin_edges = _to_distribution(ref_values, bins=bins)
    q, _         = _to_distribution(cur_values, bins=bins, bin_edges=bin_edges)
    kl = float(np.sum(rel_entr(p, q)))

    if kl < KL_WARNING_THRESHOLD:
        severity = "INFO"
    elif kl < KL_CRITICAL_THRESHOLD:
        severity = "WARNING"
    else:
        severity = "CRITICAL"

    return {
        "feature": feature,
        "kl_divergence": round(kl, 4),
        "drift_detected": kl > KL_WARNING_THRESHOLD,
        "severity": severity
    }


def kl_divergence_categorical(baseline_freq: dict, current_series, feature) -> dict:
    all_cats = sorted(set(baseline_freq.keys()) |
                      set(current_series.astype(str).unique()))
    n_base = sum(baseline_freq.values())
    n_cur  = len(current_series)

    p = np.array([baseline_freq.get(c, 1e-9) / n_base for c in all_cats])
    q_counts = np.array([(current_series.astype(str) == c).sum()
                          for c in all_cats], dtype=float)
    q = (q_counts + 1e-9) / (q_counts + 1e-9).sum()

    kl = float(np.sum(rel_entr(p, q)))

    severity = ("INFO" if kl < KL_WARNING_THRESHOLD
                else "CRITICAL" if kl > KL_CRITICAL_THRESHOLD
                else "WARNING")

    return {
        "feature": feature,
        "kl_divergence": round(kl, 4),
        "drift_detected": kl > KL_WARNING_THRESHOLD,
        "severity": severity
    }


def run_kl_drift(baseline_stats: dict, current_df,
                 numerical_features: list, categorical_features: list) -> dict:
    results = {}
    for f in numerical_features:
        if f not in current_df.columns or f not in baseline_stats.get("numerical", {}):
            continue
        ref_vals = baseline_stats["numerical"][f]["values"]
        results[f] = kl_divergence_numerical(ref_vals, current_df[f].tolist(), f)

    for f in categorical_features:
        if f not in current_df.columns or f not in baseline_stats.get("categorical", {}):
            continue
        results[f] = kl_divergence_categorical(
            baseline_stats["categorical"][f], current_df[f], f
        )
    return results