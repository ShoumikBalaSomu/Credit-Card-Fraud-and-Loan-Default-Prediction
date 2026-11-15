import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier


def test_shap_local_accuracy_property():
    """Verify Shapley efficiency property: the sum of SHAP attributions plus the expected value

    (base_value) exactly equals the model's raw output.
    """
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(200, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.5).astype(int)

    rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    sample = X[:5]
    shap_vals = explainer.shap_values(sample)

    # In shap, for binary RF, shap_values can be a list of 2 arrays [class_0, class_1] or a 3D array (N, features, 2)
    if isinstance(shap_vals, list):
        shap_class1 = shap_vals[1]
        base_val = explainer.expected_value[1]
    elif shap_vals.ndim == 3:
        shap_class1 = shap_vals[:, :, 1]
        base_val = (
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )
    else:
        shap_class1 = shap_vals
        base_val = explainer.expected_value

    preds = rf.predict_proba(sample)[:, 1]

    # Verify for each instance: sum(phi_i) + base_val == pred
    for i in range(len(sample)):
        sum_shap = np.sum(shap_class1[i]) + base_val
        assert np.isclose(
            sum_shap, preds[i], atol=1e-3
        ), f"Local accuracy violated: sum={sum_shap}, pred={preds[i]}"
