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
    )
    summary = bench.run(X=X_raw, y=y, cv=cv)

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
    )
    summary = bench.run(X=X_raw, y=y, cv=cv)

    assert "train" in summary.columns
    assert "test" in summary.columns
    assert (reg.__class__.__name__, "mean_squared_error") in summary.index


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_benchmarking_reusability(aptamer_seq, protein_seq):
    """
    Test that the same Benchmarking instance can be reused across different datasets.
    """
    clf = AptaNetPipeline()
    
    bench = Benchmarking(
        estimators=[clf],
        metrics=[accuracy_score],
    )
    
    # First dataset
    X1 = [(aptamer_seq, protein_seq) for _ in range(40)]
    y1 = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    test_fold1 = np.ones(len(y1), dtype=int) * -1
    test_fold1[-2:] = 0
    cv1 = PredefinedSplit(test_fold1)
    
    summary1 = bench.run(X=X1, y=y1, cv=cv1)
    assert summary1 is not None
    
    # Second dataset (reusing the same bench instance)
    X2 = [(aptamer_seq, protein_seq) for _ in range(30)]
    y2 = np.array([1] * 15 + [0] * 15, dtype=np.float32)
    test_fold2 = np.ones(len(y2), dtype=int) * -1
    test_fold2[-3:] = 0
    cv2 = PredefinedSplit(test_fold2)
    
    summary2 = bench.run(X=X2, y=y2, cv=cv2)
    assert summary2 is not None
    assert summary1 is not summary2  # Different results


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_benchmarking_string_metrics(aptamer_seq, protein_seq):
    """
    Test Benchmarking with string-based metric names.
    """
    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    clf = AptaNetPipeline()

    test_fold = np.ones(len(y), dtype=int) * -1
    test_fold[-2:] = 0
    cv = PredefinedSplit(test_fold)

    bench = Benchmarking(
        estimators=[clf],
        metrics=["accuracy"],  # String-based metric
    )
    summary = bench.run(X=X_raw, y=y, cv=cv)

    assert "train" in summary.columns
    assert "test" in summary.columns
    assert (clf.__class__.__name__, "accuracy") in summary.index


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_benchmarking_estimator_name_collision(aptamer_seq, protein_seq):
    """
    Test that multiple instances of the same estimator class get unique names.
    """
    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    clf1 = AptaNetPipeline(k=3)
    clf2 = AptaNetPipeline(k=4)
    clf3 = AptaNetPipeline(k=5)

    test_fold = np.ones(len(y), dtype=int) * -1
    test_fold[-2:] = 0
    cv = PredefinedSplit(test_fold)

    bench = Benchmarking(
        estimators=[clf1, clf2, clf3],
        metrics=[accuracy_score],
    )
    summary = bench.run(X=X_raw, y=y, cv=cv)

    # Check that all three estimators have unique names
    estimator_names = summary.index.get_level_values("estimator").unique()
    assert len(estimator_names) == 3
    assert "AptaNetPipeline" in estimator_names
    assert "AptaNetPipeline_1" in estimator_names
    assert "AptaNetPipeline_2" in estimator_names


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_benchmarking_custom_estimator_names(aptamer_seq, protein_seq):
    """
    Test Benchmarking with custom estimator names using (name, estimator) tuples.
    """
    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    clf1 = AptaNetPipeline(k=3)
    clf2 = AptaNetPipeline(k=4)

    test_fold = np.ones(len(y), dtype=int) * -1
    test_fold[-2:] = 0
    cv = PredefinedSplit(test_fold)

    bench = Benchmarking(
        estimators=[("k3_model", clf1), ("k4_model", clf2)],
        metrics=[accuracy_score],
    )
    summary = bench.run(X=X_raw, y=y, cv=cv)

    # Check that custom names are used
    estimator_names = summary.index.get_level_values("estimator").unique()
    assert "k3_model" in estimator_names
    assert "k4_model" in estimator_names
