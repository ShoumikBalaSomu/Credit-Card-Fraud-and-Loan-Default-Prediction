from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_disparate_impact_ratio(
    y_pred: np.ndarray, sensitive_attribute: np.ndarray, positive_is_adverse: bool = True
) -> float:
    """Compute Disparate Impact Ratio according to the EEOC 4/5ths (80%) rule.

    Parameters
    ----------
    y_pred : np.ndarray
        Model binary predictions.
    sensitive_attribute : np.ndarray
        Binary or categorical protected attribute array.
    positive_is_adverse : bool, default=True
        In risk prediction (fraud/default), positive label (1) is adverse (denial/block).
        Thus favorable outcome is y_pred == 0 (approval/allow).

    Returns
    -------
    float
        Ratio of minimum group favorable rate to maximum group favorable rate.
    """
    favorable = (y_pred == 0) if positive_is_adverse else (y_pred == 1)
    unique_groups = np.unique(sensitive_attribute)
    if len(unique_groups) < 2:
        return 1.0

    rates = []
    for g in unique_groups:
        mask = sensitive_attribute == g
        if np.sum(mask) > 0:
            rates.append(float(np.mean(favorable[mask])))

    min_rate = min(rates)
    max_rate = max(rates)
    if max_rate == 0.0:
        return 1.0
    return float(min_rate / max_rate)


def compute_equal_opportunity_diff(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive_attribute: np.ndarray
) -> float:
    """Compute maximum difference in True Positive Rate (Recall) across groups."""
    unique_groups = np.unique(sensitive_attribute)
    recalls = []
    for g in unique_groups:
        mask = (sensitive_attribute == g) & (y_true == 1)
        if np.sum(mask) > 0:
            recalls.append(float(np.mean(y_pred[mask] == 1)))
        else:
            recalls.append(0.0)

    if len(recalls) < 2:
        return 0.0
    return float(max(recalls) - min(recalls))


def audit_subgroup_fairness(
    df: pd.DataFrame,
    sensitive_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_is_adverse: bool = True,
) -> Dict[str, Any]:
    """Generate comprehensive subgroup fairness audit report."""
    subgroups = df[sensitive_col].to_numpy()
    unique_vals = np.unique(subgroups)

    group_metrics = {}
    favorable = (y_pred == 0) if positive_is_adverse else (y_pred == 1)

    for val in unique_vals:
        mask = subgroups == val
        size = int(np.sum(mask))
        fav_rate = float(np.mean(favorable[mask])) if size > 0 else 0.0

        pos_mask = mask & (y_true == 1)
        tpr = float(np.mean(y_pred[pos_mask] == 1)) if np.sum(pos_mask) > 0 else 0.0

        neg_mask = mask & (y_true == 0)
        fpr = float(np.mean(y_pred[neg_mask] == 1)) if np.sum(neg_mask) > 0 else 0.0

        group_metrics[str(val)] = {
            "sample_count": size,
            "favorable_rate": fav_rate,
            "tpr_recall": tpr,
            "fpr": fpr,
        }

    di_ratio = compute_disparate_impact_ratio(
        y_pred, subgroups, positive_is_adverse=positive_is_adverse
    )
    eod = compute_equal_opportunity_diff(y_true, y_pred, subgroups)

    return {
        "sensitive_column": sensitive_col,
        "disparate_impact_ratio": di_ratio,
        "passes_four_fifths_rule": bool(di_ratio >= 0.80),
        "equal_opportunity_diff": eod,
        "group_metrics": group_metrics,
    }
