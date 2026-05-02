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


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_benchmarking_return_raw(aptamer_seq, protein_seq):
    """
    Test that Benchmarking.run(return_raw=True) returns both the summary and raw DataFrame.
    """
    X_raw = [(aptamer_seq, protein_seq) for _ in range(20)]
    y = np.array([0] * 10 + [1] * 10, dtype=np.float32)

    clf = AptaNetPipeline()
    cv = PredefinedSplit(np.ones(len(y), dtype=int) * 0)

    bench = Benchmarking(
        estimators=[clf],
        metrics=[accuracy_score],
        X=X_raw,
        y=y,
        cv=cv,
    )
    summary, raw = bench.run(return_raw=True)

    # Check summary
    assert "train" in summary.columns
    assert "test" in summary.columns
    
    # Check raw
    assert "train" in raw.columns
    assert "test" in raw.columns
    assert raw.index.names == ["estimator", "metric", "fold"]
    assert len(raw) > 0


def test_benchmarking_labels():
    """
    Test that Benchmarking validates the length of labels matches estimators.
    """
    clf1 = AptaNetPipeline()
    clf2 = AptaNetPipeline()
    
    # Passing 2 estimators but only 1 label should raise ValueError
    with pytest.raises(ValueError, match="Length of labels must match length of estimators"):
        bench = Benchmarking(
            estimators=[clf1, clf2],
            metrics=[accuracy_score],
            X=[("A", "B")],
            y=np.array([1]),
            cv=None,
            labels=["OnlyOneLabel"]
        )
        bench.run()
