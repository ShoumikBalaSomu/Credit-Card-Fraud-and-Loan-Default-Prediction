import numpy as np


def compute_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Compute Population Stability Index (PSI) between two continuous distributions.

    PSI < 0.1: No significant change / stable
    0.1 <= PSI < 0.2: Moderate drift
    PSI >= 0.2: Significant drift / recalibration required
    """
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(expected, percentiles)
    bucket_edges[0] -= 1e-5
    bucket_edges[-1] += 1e-5

    expected_counts = np.histogram(expected, bins=bucket_edges)[0]
    actual_counts = np.histogram(actual, bins=bucket_edges)[0]

    expected_pct = np.clip(expected_counts / len(expected), 1e-4, 1.0)
    actual_pct = np.clip(actual_counts / len(actual), 1e-4, 1.0)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def test_psi_identical_distributions_yields_zero():
    np.random.seed(42)
    dist1 = np.random.normal(loc=100.0, scale=15.0, size=5000)
    dist2 = np.random.normal(loc=100.0, scale=15.0, size=5000)

    psi = compute_psi(dist1, dist2)
    assert psi < 0.05, f"Expected near-zero PSI for identical distributions, got {psi}"


def test_psi_detects_distributional_drift():
    np.random.seed(42)
    baseline_amounts = np.random.exponential(scale=50.0, size=5000)
    # Severe inflation / economic shift doubles average amount
    shifted_amounts = np.random.exponential(scale=150.0, size=5000)

    psi = compute_psi(baseline_amounts, shifted_amounts)
    assert psi > 0.25, f"PSI should flag significant drift (>0.25), got {psi}"
