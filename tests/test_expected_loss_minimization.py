import numpy as np

from src.evaluate.metrics import compute_all_metrics
from src.models.cost_sensitive import CostMatrix, ThresholdOptimizer


def test_expected_loss_strictly_beats_naive_default():
    np.random.seed(42)
    n = 2000
    y_true = np.random.binomial(1, 0.02, size=n)  # 2% rare fraud
    # Simulated model with moderate discrimination
    y_proba = np.clip(y_true * 0.35 + np.random.beta(1, 10, size=n), 0.0, 1.0)

    # Financial cost matrix: FN is 40x more expensive than FP
    cm = CostMatrix(cost_fn=200.0, cost_fp=5.0)

    optimizer = ThresholdOptimizer(cost_matrix=cm, n_thresholds=100)
    optimizer.fit(y_true, y_proba)
    p_star = optimizer.optimal_threshold_

    # Compute metrics at default 0.50 vs p*
    m_default = compute_all_metrics(y_true, y_proba, threshold=0.50, cost_matrix=cm)
    m_optimal = compute_all_metrics(y_true, y_proba, threshold=p_star, cost_matrix=cm)

    assert m_optimal["total_cost"] < m_default["total_cost"]
    assert m_optimal["cost_savings_pct"] > 0.0
    # Optimal threshold should be shifted toward the rare positive class
    assert p_star < 0.50


def test_loss_monotonicity_under_cost_scaling():
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.05, size=500)
    y_proba = np.random.uniform(0.0, 1.0, size=500)

    cm_base = CostMatrix(cost_fn=100.0, cost_fp=10.0)
    cm_scaled = CostMatrix(cost_fn=200.0, cost_fp=20.0)

    opt_base = ThresholdOptimizer(cost_matrix=cm_base, n_thresholds=50)
    opt_scaled = ThresholdOptimizer(cost_matrix=cm_scaled, n_thresholds=50)

    opt_base.fit(y_true, y_proba)
    opt_scaled.fit(y_true, y_proba)

    # Scaling all costs by factor 2 should preserve the exact optimal threshold
    assert np.isclose(opt_base.optimal_threshold_, opt_scaled.optimal_threshold_, atol=0.05)
    # And total cost should double
    assert np.isclose(opt_scaled.min_cost_, 2.0 * opt_base.min_cost_, rtol=0.05)
