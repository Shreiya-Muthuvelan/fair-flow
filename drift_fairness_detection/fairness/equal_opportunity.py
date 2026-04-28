# drift_fairness_detection/fairness/equal_opportunity.py
import pandas as pd
import numpy as np
from drift_fairness_detection.config import EQUAL_OPP_THRESHOLD

def monitor_equal_opportunity(y_pred, y_true, sensitive_df: pd.DataFrame) -> dict:
    results = {}
    pred  = pd.Series(y_pred,  index=sensitive_df.index)
    truth = pd.Series(y_true,  index=sensitive_df.index)

    for attr in sensitive_df.columns:
        tpr_by_group = {}
        for grp, idx in sensitive_df.groupby(attr).groups.items():
            qualified = truth.loc[idx] == 1   # truly good credit
            if qualified.sum() == 0:
                tpr_by_group[str(grp)] = None
                continue
            tpr_by_group[str(grp)] = round(
                float(pred.loc[idx][qualified].mean()), 4
            )

        valid = [v for v in tpr_by_group.values() if v is not None]
        diff  = round(max(valid) - min(valid), 4) if len(valid) >= 2 else 0.0

        results[attr] = {
            "tpr_by_group": tpr_by_group,
            "equal_opportunity_difference": diff,
            "violation": diff > EQUAL_OPP_THRESHOLD,
            "severity": "CRITICAL" if diff > EQUAL_OPP_THRESHOLD else "INFO"
        }

    overall = any(r["violation"] for r in results.values())
    return {"by_attribute": results, "overall_violation": overall}