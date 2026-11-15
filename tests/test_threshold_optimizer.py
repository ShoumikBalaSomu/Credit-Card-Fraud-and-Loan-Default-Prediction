import numpy as np

from src.models.cost_sensitive import CostMatrix, ThresholdOptimizer


def test_threshold_optimizer_perfect_separation():
    # Setup perfect separation
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9, 0.95])

    cm = CostMatrix(cost_fn=200.0, cost_fp=5.0)
    optimizer = ThresholdOptimizer(cost_matrix=cm, n_thresholds=100)
    optimizer.fit(y_true, y_proba)

    assert 0.25 < optimizer.optimal_threshold_ < 0.75
    assert optimizer.min_cost_ == 0.0

    preds = optimizer.predict(y_proba)
    assert np.array_equal(preds, y_true)


def test_threshold_optimizer_extreme_cost_ratio_lowers_threshold():
    # High FN cost should force threshold down to maximize recall
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.20, size=n)
    # Positives skewed high, negatives skewed low with overlapping support across [0, 1]
    y_proba = np.where(
        y_true == 1,
        np.random.beta(5, 2, size=n),
        np.random.beta(2, 5, size=n),
    )

    # C(FN) = 500, C(FP) = 1 (Missing a positive is catastrophic)
    cm_high_fn = CostMatrix(cost_fn=500.0, cost_fp=1.0)
    opt_fn = ThresholdOptimizer(cost_matrix=cm_high_fn, n_thresholds=100)
    opt_fn.fit(y_true, y_proba)

    # C(FN) = 1, C(FP) = 500 (False alarm is catastrophic)
    cm_high_fp = CostMatrix(cost_fn=1.0, cost_fp=500.0)
    opt_fp = ThresholdOptimizer(cost_matrix=cm_high_fp, n_thresholds=100)
    opt_fp.fit(y_true, y_proba)

    assert opt_fn.optimal_threshold_ < opt_fp.optimal_threshold_
    assert opt_fn.optimal_threshold_ < 0.35
    assert opt_fp.optimal_threshold_ > 0.65


def test_threshold_optimizer_predict_output_contract():
    y_true = np.array([0, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.8, 0.4, 0.6, 0.2])

    cm = CostMatrix(cost_fn=100.0, cost_fp=10.0)
    opt = ThresholdOptimizer(cost_matrix=cm)
    opt.fit(y_true, y_proba)

    preds = opt.predict(y_proba)
    assert preds.shape == y_proba.shape
    assert preds.dtype in (np.int32, np.int64, int)
    assert set(np.unique(preds)).issubset({0, 1})
