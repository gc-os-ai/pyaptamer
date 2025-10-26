import sys

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import PredefinedSplit

from pyaptamer.aptanet import AptaNetPipeline, AptaNetRegressor
from pyaptamer.benchmarking._base import Benchmarking

params = [
    (
        "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    )
]


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="skorch does not support Python 3.13"
)
@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_benchmarking_with_predefined_split_classification(aptamer_seq, protein_seq):
    """
    Test Benchmarking on a classification task using PredefinedSplit.
    """
    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    clf = AptaNetPipeline()

    test_fold = np.ones(len(y), dtype=int) * -1
    test_fold[-2:] = 0
    cv = PredefinedSplit(test_fold)

    bench = Benchmarking(
        estimators=[clf],
        metrics=[accuracy_score],
        X=X_raw,
        y=y,
        cv=cv,
    )
    summary = bench.run()

    assert "train" in summary.columns
    assert "test" in summary.columns
    assert (clf.__class__.__name__, "accuracy_score") in summary.index


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="skorch does not support Python 3.13"
)
@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_benchmarking_with_predefined_split_regression(aptamer_seq, protein_seq):
    """
    Test Benchmarking on a regression task using PredefinedSplit.
    """
    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.linspace(0, 1, 40).astype(np.float32)

    reg = AptaNetPipeline(estimator=AptaNetRegressor())

    test_fold = np.ones(len(y), dtype=int) * -1
    test_fold[-3:] = 0
    cv = PredefinedSplit(test_fold)

    bench = Benchmarking(
        estimators=[reg],
        metrics=[mean_squared_error],
        X=X_raw,
        y=y,
        cv=cv,
    )
    summary = bench.run()

    assert "train" in summary.columns
    assert "test" in summary.columns
    assert (reg.__class__.__name__, "mean_squared_error") in summary.index
