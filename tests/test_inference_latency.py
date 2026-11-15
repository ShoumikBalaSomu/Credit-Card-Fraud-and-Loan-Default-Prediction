import time
from pathlib import Path

import joblib
import numpy as np
import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "results"


def test_fraud_model_inference_latency_sla():
    model_path = RESULTS_DIR / "best_model_fraud.joblib"
    prep_path = RESULTS_DIR / "preprocessor_fraud.joblib"
    if not model_path.exists() or not prep_path.exists():
        pytest.skip("Serialized model or preprocessor not found")

    model = joblib.load(model_path)
    prep = joblib.load(prep_path)
    n_features = len(prep.get_feature_names_out())
    sample = np.random.randn(1, n_features)

    # Warmup
    for _ in range(10):
        _ = model.predict_proba(sample)

    # Benchmark 50 single-sample inferences
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model.predict_proba(sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    avg_latency = float(np.mean(latencies))
    # Sub-10ms authorization SLA requirement
    assert avg_latency < 10.0, f"Average latency {avg_latency:.4f} ms exceeded 10ms SLA"


def test_loan_model_inference_latency_sla():
    model_path = RESULTS_DIR / "best_model_loan.joblib"
    prep_path = RESULTS_DIR / "preprocessor_loan.joblib"
    if not model_path.exists() or not prep_path.exists():
        pytest.skip("Serialized model or preprocessor not found")

    model = joblib.load(model_path)
    prep = joblib.load(prep_path)
    n_features = len(prep.get_feature_names_out())
    sample = np.random.randn(1, n_features)

    # Warmup
    for _ in range(5):
        _ = model.predict_proba(sample)

    t0 = time.perf_counter()
    _ = model.predict_proba(sample)
    latency = (time.perf_counter() - t0) * 1000.0

    assert latency < 50.0, f"Latency {latency:.4f} ms exceeded batch threshold"
