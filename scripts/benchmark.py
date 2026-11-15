import os
import time
from pathlib import Path

import pandas as pd

from src.data.load import load_fraud_raw, load_loan_raw
from src.data.preprocess import preprocess_fraud_df, preprocess_loan_df
from src.utils.artifacts import load_artifact


def benchmark_pipeline():
    print("=== RUNNING INFERENCE LATENCY & MODEL BENCHMARK ===")
    benchmarks = []

    for task in ["fraud", "loan"]:
        model_path = Path(f"reports/results/best_model_{task}.joblib")
        if not model_path.exists():
            continue
        model = load_artifact(model_path)
        prep = load_artifact(f"reports/results/preprocessor_{task}.joblib")
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        if task == "fraud":
            _, df_raw = load_fraud_raw(max_train_rows=10, max_test_rows=1000)
            df = preprocess_fraud_df(df_raw)
        else:
            df_raw = load_loan_raw(max_rows=1000)
            df = preprocess_loan_df(df_raw)

        X = prep.transform(df)

        # Measure latency for 1000 rows
        t0 = time.time()
        for _ in range(10):
            _ = model.predict_proba(X)
        total_time = (time.time() - t0) / 10.0
        latency_per_sample_ms = (total_time / len(X)) * 1000.0
        throughput_samples_sec = len(X) / total_time

        benchmarks.append(
            {
                "Task": task.upper(),
                "Model File Size (MB)": round(model_size_mb, 2),
                "Latency per sample (ms)": round(latency_per_sample_ms, 4),
                "Throughput (samples/sec)": int(throughput_samples_sec),
            }
        )

    df_b = pd.DataFrame(benchmarks)
    print(df_b.to_string(index=False))
    df_b.to_csv("reports/tables/system_benchmarks.csv", index=False)
    print("Saved to reports/tables/system_benchmarks.csv")


if __name__ == "__main__":
    benchmark_pipeline()
