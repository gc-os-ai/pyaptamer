__author__ = ["nennomp", "satvshr"]


import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptanet import AptaNetClassifier, AptaNetPipeline, AptaNetRegressor

params = [
    (
        "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    )
]


class _DummyFitPipeline:
    def __init__(self):
        self.fit_args = None

    def fit(self, X, y):
        self.fit_args = (X, y)
        return self


class _DummyClassifierPipeline:
    def __init__(self, expected):
        self.expected = expected

    def predict_proba(self, X):
        return np.repeat(self.expected[None, :], len(X), axis=0)


class _DummyRegressorPipeline:
    def predict(self, X):
        return np.zeros(len(X), dtype=np.float32)


def test_pipeline_fit_returns_self(monkeypatch):
    """AptaNetPipeline.fit should follow the sklearn contract and return self."""
    pipe = AptaNetPipeline()
    dummy_pipeline = _DummyFitPipeline()

    monkeypatch.setattr(AptaNetPipeline, "_build_pipeline", lambda self: dummy_pipeline)

    X_raw = [("ACGU", "ACDE")]
    y = np.array([1], dtype=np.float32)

    result = pipe.fit(X_raw, y)

    assert result is pipe
    assert pipe.pipeline_ is dummy_pipeline
    assert dummy_pipeline.fit_args == (X_raw, y)


def test_pipeline_predict_proba_delegates_for_classifier_pipeline():
    """predict_proba should still delegate normally for classifier-backed pipelines."""
    pipe = AptaNetPipeline()
    pipe.pipeline_ = _DummyClassifierPipeline(expected=np.array([0.25, 0.75]))
    pipe._estimator = AptaNetClassifier()

    proba = pipe.predict_proba([("ACGU", "ACDE"), ("UGCA", "WXYZ")])

    assert proba.shape == (2, 2)
    assert np.allclose(proba, np.array([[0.25, 0.75], [0.25, 0.75]]))


def test_pipeline_predict_proba_raises_clear_error_for_regressor():
    """predict_proba should fail with a clear message for regressor-backed pipelines."""
    pipe = AptaNetPipeline(estimator=AptaNetRegressor())
    pipe.pipeline_ = _DummyRegressorPipeline()
    pipe._estimator = AptaNetRegressor()

    with pytest.raises(
        AttributeError,
        match="only available when the wrapped estimator implements predict_proba",
    ):
        pipe.predict_proba([("ACGU", "ACDE")])


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_and_predict_classification(aptamer_seq, protein_seq):
    """
    Test if Pipeline predictions are valid class labels and shape matches input
    for classification.
    """
    pipe = AptaNetPipeline(k=4)

    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X_raw, y)
    preds = pipe.predict(X_raw)

    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_and_predict_proba(aptamer_seq, protein_seq):
    """
    Test if Pipeline probability estimates predictions returns floats and shape matches
    input.
    """
    pipe = AptaNetPipeline()

    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X_raw, y)
    preds = pipe.predict_proba(X_raw)

    assert preds.shape == (40, 2)
    assert preds.dtype == np.float32
    assert np.all((preds >= 0) & (preds <= 1))


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_and_predict_regression(aptamer_seq, protein_seq):
    """
    Test if Pipeline predictions are valid floats and shape matches input
    for regression.
    """
    pipe = AptaNetPipeline(estimator=AptaNetRegressor())

    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.linspace(0, 1, 40).astype(np.float32)

    pipe.fit(X_raw, y)
    preds = pipe.predict(X_raw)

    assert preds.shape == (40,)
    assert np.issubdtype(preds.dtype, np.floating)


@parametrize_with_checks(
    estimators=[AptaNetClassifier(), AptaNetRegressor()],
    expected_failed_checks={
        "check_pipeline_consistency": "estimator is non-deterministic"
    },
)
def test_sklearn_compatible_estimator(estimator, check):
    """
    Run scikit-learn's compatibility checks on the AptaNetClassifier.
    """
    check(estimator)
