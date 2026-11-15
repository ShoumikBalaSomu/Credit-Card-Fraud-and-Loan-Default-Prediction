from typing import Any, Dict

import numpy as np
from sklearn.calibration import calibration_curve


def compute_calibration_stats(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    return {"ece": ece, "prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
