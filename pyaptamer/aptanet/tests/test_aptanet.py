__author__ = ["nennomp", "satvshr"]


import sys

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptanet import AptaNetClassifier, AptaNetPipeline


@pytest.fixture
def aptamer_seq():
    return "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"


@pytest.fixture
def protein_seq():
    return "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="skorch does not support Python 3.13"
)
def test_pipeline_fit_and_predict(aptamer_seq, protein_seq):
    """
    Test if Pipeline predictions are valid class labels and shape matches input.
    """
    pipe = AptaNetPipeline()

    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X_raw, y)
    preds = pipe.predict(X_raw)

    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="skorch does not support Python 3.13"
)
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


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="skorch does not support Python 3.13"
)
@parametrize_with_checks(
    estimators=[AptaNetClassifier()],
    # TODO: for some reason, despite including `check_pipeline_consistency` in the
    # checks that are supposed to fail (via `expected_failed_checks` parameter), the
    # check is still run and obviously fails. Currently, the if block is the only
    # workaround that works, and skips the check completely. Note that,
    # `check_pipeline_consistency` will never pass for non-deterministic estimators as
    # in our case. If anyone has a better working solution, please suggest.
    # expected_failed_checks={
    #    "check_pipeline_consistency": "estimator is non-deterministic"
    # },
)
def test_sklearn_compatible_estimator(estimator, check):
    """
    Run scikit-learn's compatibility checks on the AptaNetClassifier.
    """
    expected_failed_checks = ["check_pipeline_consistency"]
    if check.func.__name__ not in expected_failed_checks:
        try:
            check(estimator)
        except Exception as e:
            pytest.fail(f"Estimator check failed: {e}")
