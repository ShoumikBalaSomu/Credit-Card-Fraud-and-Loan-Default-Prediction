from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_logistic_regression(
    class_weight=None, random_state: int = 42, **kwargs
) -> LogisticRegression:
    params = dict(
        max_iter=1000, class_weight=class_weight, random_state=random_state, solver="lbfgs"
    )
    params.update(kwargs)
    return LogisticRegression(**params)


def get_random_forest(
    class_weight=None, n_estimators: int = 100, random_state: int = 42, **kwargs
) -> RandomForestClassifier:
    params = dict(
        n_estimators=n_estimators,
        max_depth=12,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)
