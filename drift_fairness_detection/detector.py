# drift_fairness_detection/detector.py
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from drift_fairness_detection.config import (BASELINE_PATH, MODEL_PATH, MODEL_FEATURES,
                            DECISION_THRESHOLD, NUMERICAL_DRIFT_FEATURES,
                            CATEGORICAL_DRIFT_FEATURES, FAIRNESS_ATTRS)
from drift_fairness_detection.drift.ks_test          import run_ks_drift
from drift_fairness_detection.drift.chi_square       import run_chi_square_drift
from drift_fairness_detection.drift.kl_divergence    import run_kl_drift
from drift_fairness_detection.drift.prediction_drift import monitor_prediction_drift
from drift_fairness_detection.fairness.demographic_parity import monitor_demographic_parity
from drift_fairness_detection.fairness.equal_opportunity  import monitor_equal_opportunity
from drift_fairness_detection.fairness.disparate_impact   import monitor_disparate_impact
from drift_fairness_detection.logger import log_batch_metrics

class DriftBiasDetector:

    def __init__(self):
        with open(BASELINE_PATH) as f:
            self.baseline = json.load(f)
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        self.reference_X = pd.read_parquet("artifacts/splits/X_train.parquet")

    def _generate_alerts(self, data_drift, pred_drift, fairness) -> list:
        alerts = []

        # Data drift alerts
        for f, r in data_drift.get("numerical_features", {}).items():
            if r["drift_detected"] and r["severity"] in ("WARNING", "CRITICAL"):
                alerts.append({
                    "type": "DATA_DRIFT", "severity": r["severity"],
                    "feature": f,
                    "message": f"KS drift in {f}: p={r['p_value']}"
                })
        for f, r in data_drift.get("categorical_features", {}).items():
            if r["drift_detected"] and r["severity"] in ("WARNING", "CRITICAL"):
                alerts.append({
                    "type": "DATA_DRIFT", "severity": r["severity"],
                    "feature": f,
                    "message": f"Chi-square drift in {f}: p={r['p_value']}"
                })

        # Prediction drift alerts
        if pred_drift["approval_rate"]["drift_detected"]:
            alerts.append({
                "type": "PREDICTION_DRIFT", "severity": "WARNING",
                "feature": "approval_rate",
                "message": f"Approval rate shifted by {pred_drift['approval_rate']['delta']}"
            })

        # Fairness alerts — always CRITICAL
        for metric, data in fairness.items():
            if metric == "overall_violation":
                continue
            if data.get("overall_violation"):
                alerts.append({
                    "type": "FAIRNESS_VIOLATION", "severity": "CRITICAL",
                    "feature": metric,
                    "message": f"Fairness violation detected in {metric}"
                })

        return alerts

    def run(self, batch_X: pd.DataFrame, batch_y=None,
            sensitive_df: pd.DataFrame = None,
            batch_id: str = None) -> dict:

        ts = datetime.now().isoformat()
        bid = batch_id or f"batch_{ts}"

        # ── predictions ───────────────────────────────────────────────
        y_prob = self.model.predict_proba(
            batch_X[MODEL_FEATURES])[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        # ── data drift ────────────────────────────────────────────────
        ks_results  = run_ks_drift(self.reference_X, batch_X,
                                    NUMERICAL_DRIFT_FEATURES)
        chi_results = run_chi_square_drift(self.baseline, batch_X,
                                            CATEGORICAL_DRIFT_FEATURES)
        kl_results  = run_kl_drift(self.baseline, batch_X,
                                    NUMERICAL_DRIFT_FEATURES,
                                    CATEGORICAL_DRIFT_FEATURES)

        data_drift = {
            "numerical_features":   ks_results,
            "categorical_features": chi_results,
            "kl_divergences":       kl_results,
            "overall_drift_detected": any(
                r["drift_detected"] for r in
                list(ks_results.values()) + list(chi_results.values())
            )
        }

        # ── prediction drift ──────────────────────────────────────────
        pred_drift = monitor_prediction_drift(y_prob, y_pred, self.baseline)

        # ── fairness ──────────────────────────────────────────────────
        fairness = {}
        if sensitive_df is not None:
            fairness["demographic_parity"] = monitor_demographic_parity(
                y_pred, sensitive_df)
            fairness["disparate_impact"] = monitor_disparate_impact(
                y_pred, sensitive_df)
            if batch_y is not None:
                fairness["equal_opportunity"] = monitor_equal_opportunity(
                    y_pred, batch_y.values, sensitive_df)
            fairness["overall_violation"] = any(
                v.get("overall_violation", False)
                for k, v in fairness.items() if k != "overall_violation"
            )

        # ── alerts ────────────────────────────────────────────────────
        alerts = self._generate_alerts(data_drift, pred_drift, fairness)

        metrics = {
            "batch_id":           bid,
            "timestamp":          ts,
            "n_samples":          len(batch_X),
            "data_drift":         data_drift,
            "prediction_drift":   pred_drift,
            "fairness_violations": fairness,
            "alerts":             alerts
        }

        log_batch_metrics(metrics)
        print(f"[{bid}] alerts={len(alerts)} | "
      f"drift={data_drift['overall_drift_detected']} | "
      f"fairness_violation={fairness.get('overall_violation', False)}")
        return metrics