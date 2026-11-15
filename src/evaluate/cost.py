import numpy as np
import pandas as pd

from src.models.cost_sensitive import CostMatrix


def evaluate_cost_curve(
    y_true: np.ndarray, y_proba: np.ndarray, cost_matrix: CostMatrix, n_points: int = 100
) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, n_points)
    records = []
    total_samples = len(y_true)
    total_positives = int(np.sum(y_true))

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        cost = cost_matrix.compute_total_cost(y_true, y_pred)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, total_positives)
        alert_rate = (tp + fp) / total_samples

        records.append(
            {
                "threshold": float(th),
                "total_cost": float(cost),
                "cost_per_instance": float(cost / total_samples),
                "precision": float(precision),
                "recall": float(recall),
                "alert_rate": float(alert_rate),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )
    return pd.DataFrame(records)
