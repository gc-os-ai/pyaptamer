import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
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


# ---- Tests for the labels parameter and auto-deduplication ----


@pytest.fixture
def simple_data():
    """Create simple classification data for label tests."""
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]] * 5)
    y = np.array([1, 0, 1, 0] * 5)
    test_fold = np.ones(len(y)) * -1
    test_fold[-4:] = 0
    cv = PredefinedSplit(test_fold)
    return X, y, cv


class TestBenchmarkingLabels:
    """Tests for the labels parameter in Benchmarking."""

    def test_single_estimator_no_labels(self, simple_data):
        """Default label should be the class name."""
        X, y, cv = simple_data
        clf = DummyClassifier(strategy="most_frequent")
        bench = Benchmarking(
            estimators=[clf], metrics=[accuracy_score], X=X, y=y, cv=cv
        )
        assert bench.labels == ["DummyClassifier"]

    def test_duplicate_class_names_auto_deduplicated(self, simple_data):
        """Two DummyClassifiers should get _1 and _2 suffixes."""
        X, y, cv = simple_data
        clf1 = DummyClassifier(strategy="most_frequent")
        clf2 = DummyClassifier(strategy="stratified")
        bench = Benchmarking(
            estimators=[clf1, clf2], metrics=[accuracy_score], X=X, y=y, cv=cv
        )
        assert bench.labels == ["DummyClassifier_1", "DummyClassifier_2"]

    def test_custom_labels_used(self, simple_data):
        """User-provided labels should be stored as-is."""
        X, y, cv = simple_data
        clf1 = DummyClassifier(strategy="most_frequent")
        clf2 = DummyClassifier(strategy="stratified")
        bench = Benchmarking(
            estimators=[clf1, clf2],
            metrics=[accuracy_score],
            X=X,
            y=y,
            cv=cv,
            labels=["freq_clf", "strat_clf"],
        )
        assert bench.labels == ["freq_clf", "strat_clf"]

    def test_custom_labels_wrong_length_raises(self, simple_data):
        """Labels with wrong length should raise ValueError."""
        X, y, cv = simple_data
        clf = DummyClassifier()
        with pytest.raises(ValueError, match="Length of `labels`"):
            Benchmarking(
                estimators=[clf],
                metrics=[accuracy_score],
                X=X,
                y=y,
                cv=cv,
                labels=["a", "b"],
            )

    def test_run_results_use_custom_labels(self, simple_data):
        """Run results should use the custom labels as index level 0."""
        X, y, cv = simple_data
        clf1 = DummyClassifier(strategy="most_frequent")
        clf2 = DummyClassifier(strategy="stratified")
        bench = Benchmarking(
            estimators=[clf1, clf2],
            metrics=[accuracy_score],
            X=X,
            y=y,
            cv=cv,
            labels=["freq", "strat"],
        )
        results = bench.run()
        est_names = results.index.get_level_values("estimator").unique().tolist()
        assert "freq" in est_names
        assert "strat" in est_names

    def test_run_auto_dedup_in_results(self, simple_data):
        """Auto-dedup labels should appear correctly in run() results."""
        X, y, cv = simple_data
        clf1 = DummyClassifier(strategy="most_frequent")
        clf2 = DummyClassifier(strategy="stratified")
        bench = Benchmarking(
            estimators=[clf1, clf2],
            metrics=[accuracy_score],
            X=X,
            y=y,
            cv=cv,
        )
        results = bench.run()
        est_names = results.index.get_level_values("estimator").unique().tolist()
        assert len(est_names) == 2
        assert "DummyClassifier_1" in est_names
        assert "DummyClassifier_2" in est_names
