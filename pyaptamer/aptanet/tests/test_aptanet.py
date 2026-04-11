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


def test_validate_data_importable():
    """Regression test for issue #365.

    ``validate_data`` was added as a standalone function only in scikit-learn
    1.6. On older versions the module must fall back to ``sklearn_compat``.
    If the compatibility shim is removed this import will raise ImportError
    on sklearn < 1.6.
    """
    import importlib

    import pyaptamer.aptanet._feature_classifier as mod

    try:
        importlib.reload(mod)
    except ImportError as exc:
        pytest.fail(
            f"_feature_classifier failed to import (issue #365 regression): {exc}"
        )

    assert hasattr(mod, "AptaNetClassifier")
    assert hasattr(mod, "AptaNetRegressor")
