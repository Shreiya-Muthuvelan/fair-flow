import pandas as pd
from drift_fairness_detection.config import DEMOGRAPHIC_PARITY_THRESHOLD, MIN_GROUP_SIZE


def monitor_demographic_parity(y_pred, sensitive_df: pd.DataFrame) -> dict:
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
                "approval_rates": group_rates,
                "group_sizes": group_sizes,
                "skipped_groups": skipped_groups,
                "demographic_parity_difference": None,
                "violation": False,
                "severity": "INFO",
                "note": f"Insufficient data — groups {skipped_groups} had < {MIN_GROUP_SIZE} samples"
            }
            continue

        rates = list(group_rates.values())
        diff  = round(max(rates) - min(rates), 4)

        results[attr] = {
            "approval_rates": group_rates,
            "group_sizes": group_sizes,
            "skipped_groups": skipped_groups,
            "demographic_parity_difference": diff,
            "violation": diff > DEMOGRAPHIC_PARITY_THRESHOLD,
            "severity": "CRITICAL" if diff > DEMOGRAPHIC_PARITY_THRESHOLD else "INFO",
            "note": (
                f"Skipped groups with < {MIN_GROUP_SIZE} samples: {skipped_groups}"
                if skipped_groups else None
            )
        }

    overall = any(r["violation"] for r in results.values())
    return {"by_attribute": results, "overall_violation": overall}