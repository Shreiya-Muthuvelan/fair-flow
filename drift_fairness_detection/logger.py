# layer4/logger.py
import json
from pathlib import Path
from datetime import datetime

LOGS_DIR = "artifacts/layer4_logs/"

def log_batch_metrics(metrics: dict):
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # One JSON file per batch
    filename = f"{LOGS_DIR}{metrics['batch_id']}.json"
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

def load_all_logs() -> list:
    """Function to get all batch results for the dashboard."""
    log_dir = Path(LOGS_DIR)
    if not log_dir.exists():
        return []
    
    logs = []
    for file in sorted(log_dir.glob("*.json")):
        with open(file) as f:
            logs.append(json.load(f))
    return logs

def load_latest_log() -> dict:
    """Returns the most recent batch result."""
    logs = load_all_logs()
    return logs[-1] if logs else {}

def get_drift_events(feature=None, severity=None) -> list:
    """Returns all drift events, optionally filtered."""
    logs = load_all_logs()
    events = []
    for log in logs:
        dd = log.get("data_drift", {})
        for feat, result in {**dd.get("numerical_features", {}),
                             **dd.get("categorical_features", {})}.items():
            if feature and feat != feature:
                continue
            if severity and result.get("severity") != severity:
                continue
            if result.get("drift_detected"):
                events.append({
                    "batch_id": log["batch_id"],
                    "timestamp": log["timestamp"],
                    "feature": feat,
                    "severity": result.get("severity"),
                    "statistic": result.get("ks_statistic") or result.get("chi2_statistic"),
                    "p_value": result.get("p_value"),
                    "kl_divergence": log["data_drift"].get(
                        "kl_divergences", {}).get(feat, {}).get("kl_divergence")
                })
    return events


def get_fairness_history(metric_type=None) -> list:
    """Returns fairness metrics across all batches for trend charts."""
    logs = load_all_logs()
    history = []
    for log in logs:
        fv = log.get("fairness_violations", {})
        for mtype, data in fv.items():
            if mtype == "overall_violation":
                continue
            if metric_type and mtype != metric_type:
                continue
            history.append({
                "batch_id": log["batch_id"],
                "timestamp": log["timestamp"],
                "metric_type": mtype,
                "by_attribute": data.get("by_attribute", {}),
                "overall_violation": data.get("overall_violation", False)
            })
    return history


def get_alert_history(severity=None, alert_type=None) -> list:
    """Returns all alerts across batches, optionally filtered."""
    logs = load_all_logs()
    alerts = []
    for log in logs:
        for alert in log.get("alerts", []):
            if severity and alert.get("severity") != severity:
                continue
            if alert_type and alert.get("type") != alert_type:
                continue
            alerts.append({
                "batch_id": log["batch_id"],
                "timestamp": log["timestamp"],
                **alert
            })
    return alerts


def get_approval_rate_trend() -> list:
    """Returns approval rate per batch — for Part 3 trend charts."""
    logs = load_all_logs()
    return [
        {
            "batch_id": log["batch_id"],
            "timestamp": log["timestamp"],
            "current_approval_rate": log["prediction_drift"]["approval_rate"]["current"],
            "baseline_approval_rate": log["prediction_drift"]["approval_rate"]["baseline"],
            "delta": log["prediction_drift"]["approval_rate"]["delta"],
        }
        for log in logs
    ]