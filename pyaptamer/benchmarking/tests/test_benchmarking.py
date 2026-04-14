import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import PredefinedSplit, cross_validate

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


def test_benchmarking_roc_auc_matches_sklearn_reference():
    """ROC-AUC should use score outputs when response_method is provided."""
    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=7,
    )
    clf = LogisticRegression(max_iter=200, random_state=7)

    bench = Benchmarking(
        estimators=[clf],
        metrics=[
            {
                "metric": roc_auc_score,
                "response_method": "predict_proba",
            }
        ],
        X=X,
        y=y,
        cv=5,
    )
    summary = bench.run()
    benchmark_auc = summary.loc[(clf.__class__.__name__, "roc_auc_score"), "test"]

    reference = cross_validate(
        clf,
        X,
        y,
        cv=5,
        scoring={"roc_auc": "roc_auc"},
        return_train_score=True,
    )
    sklearn_auc = float(np.mean(reference["test_roc_auc"]))

    assert benchmark_auc == pytest.approx(sklearn_auc, abs=1e-10)


def test_benchmarking_warns_for_score_metric_without_response_method():
    """Warn when a score-based metric is passed without response_method."""
    X, y = make_classification(
        n_samples=80,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=3,
    )
    clf = LogisticRegression(max_iter=200, random_state=3)

    bench = Benchmarking(
        estimators=[clf],
        metrics=[roc_auc_score],
        X=X,
        y=y,
        cv=3,
    )

    with pytest.warns(UserWarning, match="typically expects continuous scores"):
        bench.run()
