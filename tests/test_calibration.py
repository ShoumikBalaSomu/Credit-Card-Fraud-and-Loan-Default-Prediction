import numpy as np

from src.evaluate.calibration import compute_calibration_stats


def test_calibration_stats_perfect_classifier():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.05, 0.05, 0.05, 0.95, 0.95, 0.95])
    stats = compute_calibration_stats(y_true, y_pred, n_bins=5)

    assert "prob_true" in stats
    assert "prob_pred" in stats
    assert "ece" in stats
    assert 0.0 <= stats["ece"] <= 0.15


def test_brier_score_bounded():
    y_true = np.random.randint(0, 2, 200)
    y_pred = np.random.uniform(0, 1, 200)
    stats = compute_calibration_stats(y_true, y_pred, n_bins=10)
    assert 0.0 <= stats["ece"] <= 1.0
