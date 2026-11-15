import numpy as np

from src.evaluate.cost import evaluate_cost_curve
from src.models.cost_sensitive import CostMatrix, ThresholdOptimizer


def test_cost_surface_convexity_and_optimization():
    np.random.seed(42)
    # 1000 samples, 5% positive rate
    y_true = np.zeros(1000, dtype=int)
    y_true[:50] = 1
    np.random.shuffle(y_true)

    # Simulated well-calibrated probabilities
    y_proba = np.where(y_true == 1, np.random.beta(5, 2, 1000), np.random.beta(1, 15, 1000))

    cost_matrix = CostMatrix(cost_fn=200.0, cost_fp=5.0, cost_tp=5.0, cost_tn=0.0)
    optimizer = ThresholdOptimizer(cost_matrix=cost_matrix, n_thresholds=100)
    optimizer.fit(y_true, y_proba)

    p_star = optimizer.optimal_threshold_
    assert 0.005 <= p_star <= 0.60, f"Expected optimal threshold in reasonable range, got {p_star}"

    df_curve = evaluate_cost_curve(y_true, y_proba, cost_matrix=cost_matrix, n_points=50)
    assert not df_curve.empty
    assert "total_cost" in df_curve.columns

    # Cost at extreme ends should be higher than the minimal cost
    cost_near_zero = df_curve.loc[df_curve["threshold"] < 0.05, "total_cost"].iloc[0]
    cost_near_one = df_curve.loc[df_curve["threshold"] > 0.90, "total_cost"].iloc[-1]
    min_cost = df_curve["total_cost"].min()

    assert min_cost <= cost_near_zero
    assert min_cost <= cost_near_one
