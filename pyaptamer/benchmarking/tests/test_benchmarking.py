__author__ = ["siddharth7113"]

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import PredefinedSplit

from pyaptamer.aptanet import AptaNetPipeline, AptaNetRegressor
from pyaptamer.benchmarking._base import Benchmarking

# AptaNetPipeline is MoleculeLoader-only, but Benchmarking cross-validates X, and
# MoleculeLoader is not yet sklearn-sliceable. Tracked in #706.
_needs_sliceable_loader = pytest.mark.skip(
    reason="AptaNet + Benchmarking needs a sklearn-sliceable MoleculeLoader (#706)"
)

params = [
    (
        "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    )
]


@_needs_sliceable_loader
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


@_needs_sliceable_loader
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


# ---------- Tests for standard deviation columns ----------


class TestBenchmarkingStdScores:
    """Tests for the train_std and test_std columns."""

    def test_std_columns_present(self):
        """Results DataFrame should contain train_std and test_std columns."""
        from sklearn.dummy import DummyClassifier

        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]] * 5)
        y = np.array([1, 0, 1, 0] * 5)
        test_fold = np.ones(len(y)) * -1
        test_fold[-4:] = 0
        cv = PredefinedSplit(test_fold)

        bench = Benchmarking(
            estimators=[DummyClassifier(strategy="most_frequent")],
            metrics=[accuracy_score],
            X=X, y=y, cv=cv,
        )
        results = bench.run()

        assert "train_std" in results.columns
        assert "test_std" in results.columns

    def test_std_values_non_negative(self):
        """Standard deviation must be >= 0."""
        from sklearn.dummy import DummyClassifier

        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]] * 5)
        y = np.array([1, 0, 1, 0] * 5)
        test_fold = np.ones(len(y)) * -1
        test_fold[-4:] = 0
        cv = PredefinedSplit(test_fold)

        bench = Benchmarking(
            estimators=[DummyClassifier(strategy="most_frequent")],
            metrics=[accuracy_score],
            X=X, y=y, cv=cv,
        )
        results = bench.run()

        assert (results["train_std"] >= 0).all()
        assert (results["test_std"] >= 0).all()

    def test_std_columns_with_multiple_metrics(self):
        """Std columns should work with multiple metrics."""
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import precision_score

        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]] * 5)
        y = np.array([1, 0, 1, 0] * 5)
        test_fold = np.ones(len(y)) * -1
        test_fold[-4:] = 0
        cv = PredefinedSplit(test_fold)

        bench = Benchmarking(
            estimators=[DummyClassifier(strategy="most_frequent")],
            metrics=[accuracy_score, precision_score],
            X=X, y=y, cv=cv,
        )
        results = bench.run()

        assert results.shape[0] == 2  # 2 metrics
        assert "train_std" in results.columns
        assert "test_std" in results.columns
        assert results.shape[1] == 4  # train, test, train_std, test_std

