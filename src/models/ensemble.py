"""Ensemble and probability blending models for cost-sensitive financial risk."""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilityVotingClassifier(BaseEstimator, ClassifierMixin):
    """Weighted soft-voting ensemble classifier combining probability estimates."""

    def __init__(
        self,
        estimators: Optional[List[Tuple[str, BaseEstimator]]] = None,
        weights: Optional[List[float]] = None,
    ):
        self.estimators = estimators or []
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            fitted = est.fit(X, y)
            self.fitted_estimators_.append((name, fitted))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.fitted_estimators_:
            raise ValueError("Estimator not fitted.")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes), dtype=np.float64)

        weights = self.weights
        if weights is None:
            weights = [1.0 / len(self.fitted_estimators_)] * len(self.fitted_estimators_)
        else:
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

        for (_, est), w in zip(self.fitted_estimators_, weights):
            p = est.predict_proba(X)
            probas += w * p

        return probas

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
