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
def test_benchmarking_same_class_estimators_no_overwrite(aptamer_seq, protein_seq):
    """
    Check that passing two estimators of the same class produces two rows,
    not one. Regression test for silent result overwrite via class name key.
    """
    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    test_fold = np.ones(len(y), dtype=int) * -1
    test_fold[-2:] = 0
    cv = PredefinedSplit(test_fold)

    bench = Benchmarking(
        estimators=[AptaNetPipeline(k=3), AptaNetPipeline(k=4)],
        metrics=[accuracy_score],
        X=X_raw,
        y=y,
        cv=cv,
    )
    summary = bench.run()

    assert len(summary) == 2
    assert ("AptaNetPipeline_0", "accuracy_score") in summary.index
    assert ("AptaNetPipeline_1", "accuracy_score") in summary.index


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_benchmarking_single_estimator_name_unchanged(aptamer_seq, protein_seq):
    """
    Check that a single estimator still uses the plain class name (no index suffix).
    """
    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    test_fold = np.ones(len(y), dtype=int) * -1
    test_fold[-2:] = 0
    cv = PredefinedSplit(test_fold)

    bench = Benchmarking(
        estimators=[AptaNetPipeline(k=4)],
        metrics=[accuracy_score],
        X=X_raw,
        y=y,
        cv=cv,
    )
    summary = bench.run()

    assert len(summary) == 1
    assert ("AptaNetPipeline", "accuracy_score") in summary.index
