import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


def run_tree_shap_global(
    model, X_sample: np.ndarray, feature_names: list, save_path: str | Path, max_display: int = 15
) -> dict:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]
    importance_dict = {feature_names[i]: float(mean_abs_shap[i]) for i in top_indices}

    plt.figure(figsize=(8, 6), dpi=300)
    shap.summary_plot(
        vals, X_sample, feature_names=feature_names, max_display=max_display, show=False
    )
    plt.title("SHAP Global Feature Importance", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return importance_dict
