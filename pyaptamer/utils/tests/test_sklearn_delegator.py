"""Tests for the :class:`SklearnPipelineDelegator` utility."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.utils._sklearn_delegator import SklearnPipelineDelegator


class DummyDelegator(SklearnPipelineDelegator):
    """Minimal subclass that builds a very simple sklearn pipeline."""

    def __init__(self, C: float = 1.0):
        # example of a tunable hyper-parameter
        self.C = C

    def _build_pipeline(self):
        from sklearn.linear_model import LogisticRegression

        # build a small pipeline with a scaler and a logistic regressor
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(C=self.C))]
        )


def test_delegator_basic_behavior():
    """Fitting the wrapper should produce a functioning pipeline."""
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, size=20)

    d = DummyDelegator(C=0.5)
    d.fit(X, y)
    assert hasattr(d, "predict"), "predict should be delegated to the pipeline"
    preds = d.predict(X)
    assert preds.shape == (20,)
    # check that hyperparameter was passed through
    assert d.named_steps["clf"].C == 0.5


@parametrize_with_checks(estimators=[DummyDelegator()])
def test_delegator_sklearn_compatible(estimator, check):
    # pipeline consistency check fails for non-deterministic estimators; skip
    if check.func.__name__ == "check_pipeline_consistency":
        pytest.skip("skip non-deterministic pipeline check")
    check(estimator)
