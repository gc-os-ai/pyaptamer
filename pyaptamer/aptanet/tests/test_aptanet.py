__author__ = ["nennomp", "satvshr", "siddharth7113"]


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


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_with_dataframe(aptamer_seq, protein_seq):
    """Test that AptaNetPipeline accepts a pd.DataFrame input."""
    import pandas as pd

    n = 40
    df = pd.DataFrame({"aptamer": [aptamer_seq] * n, "protein": [protein_seq] * n})
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float32)

    pipe = AptaNetPipeline(k=4)
    pipe.fit(df, y)
    assert len(pipe.predict(df)) == n


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_with_numpy_pair(aptamer_seq, protein_seq):
    """Test that AptaNetPipeline accepts a (np.ndarray, np.ndarray) tuple input."""
    n = 40
    apta = np.array([aptamer_seq] * n)
    prot = np.array([protein_seq] * n)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float32)

    pipe = AptaNetPipeline(k=4)
    pipe.fit((apta, prot), y)
    assert len(pipe.predict((apta, prot))) == n


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_with_apidataset(aptamer_seq, protein_seq):
    """Test that AptaNetPipeline accepts an APIDataset input."""
    from pyaptamer.datasets.dataclasses import APIDataset

    n = 40
    pairs = [(aptamer_seq, protein_seq) for _ in range(n)]
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float32)

    ds = APIDataset.from_any(pairs, y)
    pipe = AptaNetPipeline(k=4)
    pipe.fit(ds, ds.y)
    assert len(pipe.predict(ds)) == n


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_with_dataframe_and_string_labels(aptamer_seq, protein_seq):
    """Test the real-world flow: DataFrame X + DataFrame y with string labels.

    This mimics `X, y = load_li2014(split="train"); pipeline.fit(X, y)` where
    y is a single-column DataFrame with "positive"/"negative" string labels.
    """
    import pandas as pd

    n = 40
    df = pd.DataFrame({"aptamer": [aptamer_seq] * n, "protein": [protein_seq] * n})
    y = pd.DataFrame({"label": ["positive"] * (n // 2) + ["negative"] * (n // 2)})

    pipe = AptaNetPipeline(k=4)
    pipe.fit(df, y)
    assert len(pipe.predict(df)) == n


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
