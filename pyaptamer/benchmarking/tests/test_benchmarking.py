import numpy as np
import pytest
from sklearn.metrics import accuracy_score
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
    )
    summary = bench.run(X=X_raw, y=y, cv=cv)

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
        metrics=["neg_mean_squared_error"],
    )
    summary = bench.run(X=X_raw, y=y, cv=cv)

    assert "train" in summary.columns
    assert "test" in summary.columns
    assert (reg.__class__.__name__, "neg_mean_squared_error") in summary.index


def test_benchmarking_unique_names():
    """
    Test that duplicate estimator classes get unique names.
    """
    aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    X_raw = [(aptamer_seq, protein_seq) for _ in range(20)]
    y = np.array([0] * 10 + [1] * 10, dtype=np.float32)

    clf1 = AptaNetPipeline(k=3)
    clf2 = AptaNetPipeline(k=4)

    bench = Benchmarking(
        estimators=[clf1, clf2],
        metrics=["accuracy"],
    )
    summary = bench.run(X=X_raw, y=y, cv=2)

    assert "AptaNetPipeline" in summary.index.get_level_values(0)
    assert "AptaNetPipeline_1" in summary.index.get_level_values(0)


def test_benchmarking_named_estimators():
    """
    Test that user-provided names for estimators are used.
    """
    aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    X_raw = [(aptamer_seq, protein_seq) for _ in range(20)]
    y = np.array([0] * 10 + [1] * 10, dtype=np.float32)

    clf = AptaNetPipeline(k=3)

    bench = Benchmarking(
        estimators=[("MyModel", clf)],
        metrics=["accuracy"],
    )
    summary = bench.run(X=X_raw, y=y, cv=2)

    assert "MyModel" in summary.index.get_level_values(0)
