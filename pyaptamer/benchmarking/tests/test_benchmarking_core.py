import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score

from pyaptamer.benchmarking._base import Benchmarking


def test_benchmarking_keeps_duplicate_estimator_classes_distinct():
    """Estimators with the same class should not overwrite one another."""
    X = np.array([[0], [1], [0], [1], [0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    bench = Benchmarking(
        estimators=[
            DummyClassifier(strategy="most_frequent"),
            DummyClassifier(strategy="stratified", random_state=0),
        ],
        metrics=[accuracy_score],
        X=X,
        y=y,
        cv=2,
    )

    summary = bench.run()

    assert ("DummyClassifier_1", "accuracy_score") in summary.index
    assert ("DummyClassifier_2", "accuracy_score") in summary.index
    assert len(summary) == 2


def test_benchmarking_preserves_unique_estimator_names():
    """Different estimator classes should keep their original class names."""
    bench = Benchmarking(
        estimators=[
            DummyClassifier(strategy="most_frequent"),
            DummyRegressor(strategy="mean"),
        ],
        metrics=[accuracy_score],
        X=np.array([[0], [1]]),
        y=np.array([0, 1]),
        cv=2,
    )

    names = bench._get_estimator_names()

    assert names == ["DummyClassifier", "DummyRegressor"]
