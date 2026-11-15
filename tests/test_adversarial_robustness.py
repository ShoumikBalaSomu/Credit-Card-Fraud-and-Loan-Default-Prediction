import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_model_smoothness_under_feature_jitter():
    np.random.seed(42)
    n = 1000
    p = 5
    X = np.random.normal(loc=0.0, scale=1.0, size=(n, p))
    # Simple continuous risk target
    y = ((X[:, 0] * 1.5 + X[:, 1] * 0.8 + np.random.normal(0, 0.5, size=n)) > 1.0).astype(int)

    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)

    # Base predictions on a test batch
    X_test = np.random.normal(0.0, 1.0, size=(100, p))
    base_preds = clf.predict_proba(X_test)[:, 1]

    # Add 1% epsilon jitter to continuous inputs
    epsilon = np.random.normal(0.0, 0.01, size=X_test.shape)
    perturbed_preds = clf.predict_proba(X_test + epsilon)[:, 1]

    # Mean absolute shift in probability should be small (< 0.08)
    mean_shift = np.mean(np.abs(perturbed_preds - base_preds))
    assert mean_shift < 0.08, f"Model is excessively fragile to 1% noise: shift={mean_shift}"


def test_amount_monotonicity_sanity():
    """In fraud detection, holding all else equal, massive amount increases should generally not decrease risk."""
    np.random.seed(42)
    # Synthetic fraud model trained with amount as feature index 0
    X_train = np.random.uniform(10.0, 500.0, size=(500, 3))
    # High amount strongly correlates with fraud
    y_train = (X_train[:, 0] > 300.0).astype(int)

    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)

    # Compare low vs extreme high amount
    x_low = np.array([[20.0, 0.5, 0.5]])
    x_high = np.array([[1000.0, 0.5, 0.5]])

    p_low = clf.predict_proba(x_low)[0, 1]
    p_high = clf.predict_proba(x_high)[0, 1]

    assert p_high >= p_low, "Extreme amount should not have lower risk than small amount"
