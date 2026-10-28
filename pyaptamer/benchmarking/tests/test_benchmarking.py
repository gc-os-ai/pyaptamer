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


def test_to_scorers_duplicate_metric_names_no_overwrite():
    """
    Two metrics that share the same __name__ must produce distinct scorer keys
    with _0/_1 suffixes instead of silently overwriting each other.
    """
    bench = Benchmarking(estimators=[], metrics=[], X=[], y=[])

    def metric_a(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def metric_b(y_true, y_pred):
        return accuracy_score(y_true, np.ones_like(y_pred))

    metric_b.__name__ = "metric_a"

    scorers = bench._to_scorers([metric_a, metric_b])

    assert "metric_a_0" in scorers
    assert "metric_a_1" in scorers
    assert "metric_a" not in scorers
    assert len(scorers) == 2


def test_to_scorers_single_metric_name_unchanged():
    """
    A single metric must keep its original __name__ with no suffix appended.
    """
    bench = Benchmarking(estimators=[], metrics=[], X=[], y=[])

    scorers = bench._to_scorers([accuracy_score])

    assert "accuracy_score" in scorers
    assert "accuracy_score_0" not in scorers
    assert len(scorers) == 1
