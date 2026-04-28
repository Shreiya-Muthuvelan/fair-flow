import pandas as pd
from drift_fairness_detection.config import DISPARATE_IMPACT_THRESHOLD, MIN_GROUP_SIZE


def monitor_disparate_impact(y_pred, sensitive_df: pd.DataFrame) -> dict:
    results = {}
    pred = pd.Series(y_pred, index=sensitive_df.index)

    for attr in sensitive_df.columns:
        group_rates = {}
        group_sizes = {}
        skipped_groups = []

        for grp, idx in sensitive_df.groupby(attr).groups.items():
            size = len(idx)
            group_sizes[str(grp)] = size

            if size < MIN_GROUP_SIZE:
                skipped_groups.append(str(grp))
                continue

            group_rates[str(grp)] = round(float(pred.loc[idx].mean()), 4)

        if len(group_rates) < 2:
            results[attr] = {
                "group_rates": group_rates,
                "group_sizes": group_sizes,
                "skipped_groups": skipped_groups,
                "min_group": None,
                "max_group": None,
                "disparate_impact_ratio": None,
                "violation": False,
                "severity": "INFO",
                "note": f"Insufficient data — groups {skipped_groups} had < {MIN_GROUP_SIZE} samples"
            }
            continue

        rates  = list(group_rates.values())
        groups = list(group_rates.keys())
        min_group = groups[rates.index(min(rates))]
        max_group = groups[rates.index(max(rates))]
        ratio     = round(min(rates) / max(rates), 4) if max(rates) > 0 else 1.0

        results[attr] = {
            "group_rates": group_rates,
            "group_sizes": group_sizes,
            "skipped_groups": skipped_groups,
            "min_group": min_group,
            "max_group": max_group,
            "disparate_impact_ratio": ratio,
            "violation": ratio < DISPARATE_IMPACT_THRESHOLD,
            "severity": "CRITICAL" if ratio < DISPARATE_IMPACT_THRESHOLD else "INFO",
            "note": (
                f"Skipped groups with < {MIN_GROUP_SIZE} samples: {skipped_groups}"
                if skipped_groups else None
            )
        }

    overall = any(r["violation"] for r in results.values())
    return {"by_attribute": results, "overall_violation": overall}