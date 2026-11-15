import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


def explain_instance_waterfall(
    model, instance: np.ndarray, feature_names: list, save_path: str | Path
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "fitted_estimators_"):
        model = model.fitted_estimators_[0][1]
    explainer = shap.TreeExplainer(model)
    explanation = explainer(instance.reshape(1, -1))

    if len(explanation.values.shape) == 3:
        exp_vals = explanation.values[0, :, 1]
        base_val = (
            explanation.base_values[0, 1]
            if hasattr(explanation.base_values, "__len__")
            else explanation.base_values
        )
    else:
        exp_vals = explanation.values[0]
        base_val = (
            explanation.base_values[0]
            if hasattr(explanation.base_values, "__len__")
            else explanation.base_values
        )

    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(
        shap.Explanation(
            values=exp_vals, base_values=base_val, data=instance, feature_names=feature_names
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
