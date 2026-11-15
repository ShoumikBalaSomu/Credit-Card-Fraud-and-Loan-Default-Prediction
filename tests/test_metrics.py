import numpy as np

from src.evaluate.metrics import recall_at_alert_budget, recall_at_precision


def test_recall_at_precision():
    y_true = np.array([1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
    r = recall_at_precision(y_true, y_proba, min_precision=0.8)
    assert r == 1.0


def test_recall_at_alert_budget():
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.95, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    r = recall_at_alert_budget(y_true, y_proba, budget_fraction=0.10)
    assert r == 1.0
