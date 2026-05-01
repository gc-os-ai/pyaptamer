__author__ = ["nennomp", "satvshr"]


import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptanet import AptaNetClassifier, AptaNetPipeline, AptaNetRegressor
from pyaptamer.aptanet._aptanet_nn import AptaNetMLP

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


def test_aptanet_mlp_no_input_dim_raises():
    """Check that AptaNetMLP raises ValueError when use_lazy=False and input_dim=None.

    Previously this silently passed None to nn.Linear, causing a confusing
    TypeError deep inside PyTorch with no pointer to the real cause. Fixes #611.
    """
    with pytest.raises(ValueError, match="input_dim is required when use_lazy=False"):
        AptaNetMLP(use_lazy=False, input_dim=None)