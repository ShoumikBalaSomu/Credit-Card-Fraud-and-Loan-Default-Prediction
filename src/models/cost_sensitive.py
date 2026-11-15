import numpy as np
from sklearn.metrics import confusion_matrix


class CostMatrix:
    """Cost matrix specification for binary decision tasks.

    Parameters
    ----------
    cost_fn : float
        Cost of False Negative (e.g. unrecovered fraud, loan default).
    cost_fp : float
        Cost of False Positive (e.g. customer friction, rejected good loan).
    cost_tp : float, default=0.0
        Cost of True Positive.
    cost_tn : float, default=0.0
        Cost of True Negative.
    """

    def __init__(self, cost_fn: float, cost_fp: float, cost_tp: float = 0.0, cost_tn: float = 0.0):
        self.cost_fn = float(cost_fn)
        self.cost_fp = float(cost_fp)
        self.cost_tp = float(cost_tp)
        self.cost_tn = float(cost_tn)

    def compute_total_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute total financial loss: cost_fn * FN + cost_fp * FP + cost_tp * TP + cost_tn * TN."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(self.cost_fn * fn + self.cost_fp * fp + self.cost_tp * tp + self.cost_tn * tn)

    def compute_normalized_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cost divided by sample size (expected cost per instance)."""
        return self.compute_total_cost(y_true, y_pred) / max(1, len(y_true))


class ThresholdOptimizer:
    """Finds optimal decision threshold p* minimizing total expected cost."""

    def __init__(self, cost_matrix: CostMatrix, n_thresholds: int = 200):
        self.cost_matrix = cost_matrix
        self.n_thresholds = n_thresholds
        self.optimal_threshold_ = 0.5
        self.min_cost_ = float("inf")

    def fit(self, y_true: np.ndarray, y_proba: np.ndarray) -> "ThresholdOptimizer":
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        best_thresh = 0.5
        best_cost = float("inf")

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            cost = self.cost_matrix.compute_total_cost(y_true, y_pred)
            if cost < best_cost:
                best_cost = cost
                best_thresh = thresh

        self.optimal_threshold_ = float(best_thresh)
        self.min_cost_ = float(best_cost)
        return self

    def predict(self, y_proba: np.ndarray) -> np.ndarray:
        return (y_proba >= self.optimal_threshold_).astype(int)
