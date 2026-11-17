import numpy as np
import pytest


class CostMatrix:
    """Cost matrix specification for asymmetric binary decision tasks."""

    def __init__(self, cost_fn: float, cost_fp: float, cost_tp: float = 0.0, cost_tn: float = 0.0):
        self.cost_fn = float(cost_fn)
        self.cost_fp = float(cost_fp)
        self.cost_tp = float(cost_tp)
        self.cost_tn = float(cost_tn)

    def compute_total_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return float(self.cost_fn * fn + self.cost_fp * fp + self.cost_tp * tp + self.cost_tn * tn)

    def bayes_optimal_threshold(self) -> float:
        """Computes the theoretical Bayes risk minimizing threshold: p* = C_FP / (C_FP + C_FN)."""
        num = self.cost_fp - self.cost_tn
        den = (self.cost_fp - self.cost_tn) + (self.cost_fn - self.cost_tp)
        if den == 0:
            return 0.5
        return float(np.clip(num / den, 0.001, 0.999))


class ThresholdOptimizer:
    """Finds empirical cost-minimizing threshold over calibrated posterior probabilities."""

    def __init__(self, cost_matrix: CostMatrix, n_thresholds: int = 100):
        self.cost_matrix = cost_matrix
        self.n_thresholds = n_thresholds
        self.optimal_threshold_ = 0.5
        self.min_cost_ = float("inf")

    def fit(self, y_true: np.ndarray, y_proba: np.ndarray):
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        costs = []
        for th in thresholds:
            y_pred = (y_proba >= th).astype(int)
            c = self.cost_matrix.compute_total_cost(y_true, y_pred)
            costs.append(c)

        best_idx = int(np.argmin(costs))
        self.optimal_threshold_ = float(thresholds[best_idx])
        self.min_cost_ = float(costs[best_idx])
        return self


def test_cost_matrix_calculation():
    # 40:1 cost matrix (Fraud task: C_FN = 400, C_FP = 10)
    cm = CostMatrix(cost_fn=400.0, cost_fp=10.0)

    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0])  # 1 TP, 1 FN, 1 FP, 2 TN

    total_cost = cm.compute_total_cost(y_true, y_pred)
    # Total cost = 1*400 (FN) + 1*10 (FP) = 410
    assert total_cost == 410.0


def test_bayes_optimal_threshold_formula():
    cm_fraud = CostMatrix(cost_fn=400.0, cost_fp=10.0)
    # p* = 10 / (10 + 400) = 10 / 410 = 0.02439...
    p_opt = cm_fraud.bayes_optimal_threshold()
    assert pytest.approx(p_opt, rel=1e-3) == (10.0 / 410.0)

    cm_loan = CostMatrix(cost_fn=5000.0, cost_fp=500.0)
    # p* = 500 / (500 + 5000) = 500 / 5500 = 0.0909...
    assert pytest.approx(cm_loan.bayes_optimal_threshold(), rel=1e-3) == (500.0 / 5500.0)


def test_threshold_optimizer_reduces_cost():
    cm = CostMatrix(cost_fn=400.0, cost_fp=10.0)
    y_true = np.array([0] * 90 + [1] * 10)
    # Posterior probabilities where frauds have moderate risk (e.g. 0.15 to 0.40)
    np.random.seed(42)
    y_proba = np.where(
        y_true == 1,
        np.random.uniform(0.15, 0.45, size=100),
        np.random.uniform(0.01, 0.10, size=100),
    )

    opt = ThresholdOptimizer(cost_matrix=cm, n_thresholds=100)
    opt.fit(y_true, y_proba)

    # At default p = 0.50, all 10 frauds are missed (FN = 10 -> cost = 4000)
    cost_default = cm.compute_total_cost(y_true, (y_proba >= 0.50).astype(int))
    cost_optimal = opt.min_cost_

    assert cost_default >= 4000.0
    assert cost_optimal < cost_default
    assert opt.optimal_threshold_ < 0.50
