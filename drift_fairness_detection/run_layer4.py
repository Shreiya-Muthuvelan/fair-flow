
import pandas as pd
import json
from drift_fairness_detection.detector import DriftBiasDetector

SPLITS_DIR = "artifacts/splits/"
BATCH_SIZE = 50

def run():
    detector = DriftBiasDetector()

    # Combine val + test for samples
    X_val     = pd.read_parquet(f"{SPLITS_DIR}X_val.parquet")
    X_test    = pd.read_parquet(f"{SPLITS_DIR}X_test.parquet")
    y_val     = pd.read_parquet(f"{SPLITS_DIR}y_val.parquet").squeeze()
    y_test    = pd.read_parquet(f"{SPLITS_DIR}y_test.parquet").squeeze()
    sens_val  = pd.read_parquet(f"{SPLITS_DIR}sensitive_val.parquet")
    sens_test = pd.read_parquet(f"{SPLITS_DIR}sensitive_test.parquet")

    # Concatenate and reset index — critical so groupby indices align
    X_all    = pd.concat([X_val, X_test],   ignore_index=True)
    y_all    = pd.concat([y_val, y_test],   ignore_index=True)
    sens_all = pd.concat([sens_val, sens_test], ignore_index=True)

    n_batches = len(X_all) // BATCH_SIZE
    print(f"Total rows: {len(X_all)} → {n_batches} batches of {BATCH_SIZE}\n")

    for i in range(n_batches):
        start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE

        metrics = detector.run(
            batch_X      = X_all.iloc[start:end].reset_index(drop=True),
            batch_y      = y_all.iloc[start:end].reset_index(drop=True),
            sensitive_df = sens_all.iloc[start:end].reset_index(drop=True),
            batch_id     = f"batch_{i:02d}"
        )

        n_alerts   = len(metrics["alerts"])
        drift_flag = metrics["data_drift"]["overall_drift_detected"]
        fair_flag  = metrics["fairness_violations"].get("overall_violation", False)

        print(f"[{metrics['batch_id']}] "
              f"drift={'YES' if drift_flag else 'no ':3s} | "
              f"fairness_violation={'YES' if fair_flag else 'no ':3s} | "
              f"alerts={n_alerts}")

        if metrics["alerts"]:
            for a in metrics["alerts"]:
                print(f"    [{a['severity']}] {a['type']} — {a['message']}")

    print("\nLayer 4 complete — logs saved to artifacts/layer4_logs/")

if __name__ == "__main__":
    run()