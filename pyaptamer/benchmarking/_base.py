__author__ = "satvshr"
__all__ = ["Benchmarking"]


import numpy as np
import pandas as pd
from skbase.base import BaseObject
from sklearn.model_selection import cross_validate


class Benchmarking(BaseObject):
    """
    Benchmark estimators using cross-validation.

    You can:

    - pass `X, y` (feature matrix and labels/targets) along with `cv`
      to use any cross-validation strategy;
    - if you want a fixed train/test split, pass a `PredefinedSplit`
      object as `cv`.

    Parameters
    ----------
    estimators : list[estimator] | estimator
        List of sklearn-like estimators implementing `fit` and `predict`.
    metrics : list[callable] | callable
        List of callables with signature `(y_true, y_pred) -> float`.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int, CV splitter, or None, default=None
        Cross-validation strategy. If `None`, defaults to 5-fold CV.
        If you want to use an explicit train/test split, pass a
        `PredefinedSplit` object.

    Attributes
    ----------
    results : pd.DataFrame
        Results table after calling :meth:`run`.

        - Rows = MultiIndex (estimator, metric)
        - Cols = ["train", "test"]

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.model_selection import PredefinedSplit
    >>> from pyaptamer.benchmarking._base import Benchmarking
    >>> from pyaptamer.aptanet import AptaNetPipeline
    >>> X = np.random.randn(10, 5)
    >>> y = np.random.randint(0, 2, size=10)
    >>> clf = AptaNetPipeline()
    >>> # define a fixed train/test split
    >>> test_fold = np.ones(len(y)) * -1
    >>> test_fold[-2:] = 0
    >>> cv = PredefinedSplit(test_fold)
    >>> bench = Benchmarking(
    ...     estimators=[clf],
    ...     metrics=[accuracy_score],
    ...     X=X,
    ...     y=y,
    ...     cv=cv,
    ... )
    >>> summary = bench.run()  # doctest: +SKIP
    """

    def __init__(self, estimators, metrics, X, y, cv=None):
        self.estimators = estimators if isinstance(estimators, list) else [estimators]
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.X = X
        self.y = y
        self.cv = cv
        self.results = None

    def _to_df(self, results):
        """Convert nested results to a unified DataFrame."""
        records = []
        index = []

        for est_name, est_scores in results.items():
            for metric_name, scores in est_scores.items():
                records.append(scores)
                index.append((est_name, metric_name))

        index = pd.MultiIndex.from_tuples(index, names=["estimator", "metric"])
        return pd.DataFrame(records, index=index, columns=["train", "test"])

    def run(self):
        """
        Train each estimator and evaluate with cross-validation.

        Returns
        -------
        pd.DataFrame
            Results table with rows = (estimator, metric),
            cols = ["train", "test"].
        """
        results = {}

        for estimator in self.estimators:
            est_name = estimator.__class__.__name__

            scoring = {
                getattr(
                    evaluator,
                    "__name__",
                    getattr(evaluator, "name", evaluator.__class__.__name__),
                ): evaluator
                for evaluator in self.metrics
            }

            cv_results = cross_validate(
                estimator,
                self.X,
                self.y,
                cv=self.cv,
                scoring=scoring,
                return_train_score=True,
            )

            # average across folds
            est_scores = {}
            for metric in scoring.keys():
                est_scores[metric] = {
                    "train": float(np.mean(cv_results[f"train_{metric}"])),
                    "test": float(np.mean(cv_results[f"test_{metric}"])),
                }

            results[est_name] = est_scores

        self.results = self._to_df(results)
        return self.results
