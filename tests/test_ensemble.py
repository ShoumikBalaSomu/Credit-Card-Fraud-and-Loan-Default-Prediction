import numpy as np

from src.models.baselines import get_random_forest
from src.models.ensemble import ProbabilityVotingClassifier
from src.models.gradient_boosting import get_lightgbm


def test_ensemble_voting_probability_shape_and_sum():
    np.random.seed(42)
    X = np.random.randn(80, 4)
    y = np.random.randint(0, 2, 80)

    clf1 = get_random_forest(n_estimators=10, random_state=42)
    clf2 = get_lightgbm(n_estimators=10, random_state=42)

    ens = ProbabilityVotingClassifier(
        estimators=[("rf", clf1), ("lgb", clf2)],
        weights=[0.7, 0.3],
    )
    ens.fit(X, y)
    probs = ens.predict_proba(X)

    assert probs.shape == (80, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_ensemble_predict_threshold():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    clf = get_random_forest(n_estimators=10, random_state=42)
    ens = ProbabilityVotingClassifier(estimators=[("rf", clf)])
    ens.fit(X, y)

    preds_low = ens.predict(X, threshold=0.1)
    preds_high = ens.predict(X, threshold=0.9)

    # Lower threshold should classify equal or more positives
    assert np.sum(preds_low) >= np.sum(preds_high)
