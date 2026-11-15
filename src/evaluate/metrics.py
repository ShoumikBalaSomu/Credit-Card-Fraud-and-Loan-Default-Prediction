from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.cost_sensitive import CostMatrix


def recall_at_precision(y_true: np.ndarray, y_proba: np.ndarray, min_precision: float) -> float:
    thresholds = np.linspace(0.01, 0.99, 100)
    best_recall = 0.0
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        if p >= min_precision:
            r = recall_score(y_true, y_pred, zero_division=0)
            if r > best_recall:
                best_recall = r
    return float(best_recall)


def recall_at_alert_budget(
    y_true: np.ndarray, y_proba: np.ndarray, budget_fraction: float
) -> float:
    k = max(1, int(len(y_proba) * budget_fraction))
    top_k_indices = np.argsort(y_proba)[::-1][:k]
    tp = np.sum(y_true[top_k_indices])
    total_positives = max(1, np.sum(y_true))
    return float(tp / total_positives)


def compute_all_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float, cost_matrix: CostMatrix
) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_auc = float(average_precision_score(y_true, y_proba))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_proba))

    total_cost = float(cost_matrix.compute_total_cost(y_true, y_pred))
    cost_default_05 = float(cost_matrix.compute_total_cost(y_true, (y_proba >= 0.5).astype(int)))
    cost_savings_pct = float(
        max(0.0, (cost_default_05 - total_cost) / max(1e-5, cost_default_05) * 100)
    )

    rec_at_p50 = recall_at_precision(y_true, y_proba, min_precision=0.5)
    rec_at_p80 = recall_at_precision(y_true, y_proba, min_precision=0.8)
    rec_at_top1pct = recall_at_alert_budget(y_true, y_proba, budget_fraction=0.01)
    rec_at_top2pct = recall_at_alert_budget(y_true, y_proba, budget_fraction=0.02)

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2,
        "mcc": mcc,
        "brier": brier,
        "total_cost": total_cost,
        "cost_default_05": cost_default_05,
        "cost_savings_pct": cost_savings_pct,
        "recall_at_precision_0.5": rec_at_p50,
        "recall_at_precision_0.8": rec_at_p80,
        "recall_at_alert_budget_1pct": rec_at_top1pct,
        "recall_at_alert_budget_2pct": rec_at_top2pct,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
