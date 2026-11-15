import numpy as np

from src.models.cost_sensitive import CostMatrix, ThresholdOptimizer


def test_cost_matrix_calculation():
    cm = CostMatrix(cost_fn=500.0, cost_fp=10.0, cost_tp=0.0, cost_tn=0.0)
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0])  # 1 TP, 1 FN, 1 FP, 2 TN
    total_cost = cm.compute_total_cost(y_true, y_pred)
    assert total_cost == 510.0


def test_threshold_optimizer_minimizes_cost():
    cm = CostMatrix(cost_fn=500.0, cost_fp=10.0)
    opt = ThresholdOptimizer(cost_matrix=cm, n_thresholds=100)
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.array([0.05] * 90 + [0.35] * 10)
    opt.fit(y_true, y_proba)
    preds_opt = opt.predict(y_proba)
    cost_opt = cm.compute_total_cost(y_true, preds_opt)
    cost_05 = cm.compute_total_cost(y_true, (y_proba >= 0.5).astype(int))
    assert cost_opt < cost_05
