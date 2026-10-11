__author__ = "satvshr"
__all__ = ["Benchmarking"]

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate


class Benchmarking:
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
        DataFrame produced by :meth:`run`.

        - Index: pandas.MultiIndex with two levels (names shown in parentheses)
            - level 0 "estimator": estimator name
            - level 1 "metric": evaluator name
        - Columns: ["train", "test"] (both floats)
        - Cell values: mean scores (float) computed across CV folds:
            - "train" = mean of cross_validate(...)[f"train_{metric}"]
            - "test"  = mean of cross_validate(...)[f"test_{metric}"]

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.model_selection import PredefinedSplit
    >>> from pyaptamer.benchmarking._base import Benchmarking
    >>> from pyaptamer.aptanet import AptaNetPipeline
    >>> aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> # dataset: 20 aptamer–protein pairs
    >>> X = [(aptamer_seq, protein_seq) for _ in range(20)]
    >>> y = np.array([0] * 10 + [1] * 10, dtype=np.float32)
    >>> clf = AptaNetPipeline(k=4)
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

    def __init__(self, estimators, metrics):
        self.estimators = estimators if isinstance(estimators, list) else [estimators]
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.results = None

    def _to_scorers(self, metrics):
        """Convert metrics to a dict of scorers."""
        from sklearn.metrics import get_scorer

        scorers = {}
        for metric in metrics:
            if isinstance(metric, str):
                scorers[metric] = get_scorer(metric)
            elif callable(metric):
                name = (
                    metric.__name__
                    if hasattr(metric, "__name__")
                    else metric.__class__.__name__
                )
                scorers[name] = make_scorer(metric)
            else:
                raise ValueError(f"Metric {metric} should be a callable or a string.")
        return scorers

    def _get_estimator_names(self):
        """Get or generate unique names for estimators."""
        names = []
        estimators = []

        for item in self.estimators:
            if isinstance(item, tuple) and len(item) == 2:
                names.append(item[0])
                estimators.append(item[1])
            else:
                names.append(item.__class__.__name__)
                estimators.append(item)

        # Handle duplicates by adding suffixes
        final_names = []
        counts = {}
        for name in names:
            if name not in counts:
                counts[name] = 0
                final_names.append(name)
            else:
                counts[name] += 1
                final_names.append(f"{name}_{counts[name]}")

        return list(zip(final_names, estimators, strict=False))

    def _to_df(self, results):
        """Convert nested results to a unified DataFrame."""
        records = []
        index = []

        # Ensure consistent order by iterating over the results in processing order
        for est_name, est_scores in results.items():
            for metric_name, scores in est_scores.items():
                records.append(scores)
                index.append((est_name, metric_name))

        index = pd.MultiIndex.from_tuples(index, names=["estimator", "metric"])
        return pd.DataFrame(records, index=index, columns=["train", "test"])

    def run(self, X, y, cv=None):
        """
        Train each estimator and evaluate with cross-validation.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        cv : int, CV splitter, or None, default=None
            Cross-validation strategy. If `None`, defaults to 5-fold CV.

        Returns
        -------
        results : pd.DataFrame
        """
        self.scorers_ = self._to_scorers(self.metrics)
        named_estimators = self._get_estimator_names()
        results = {}

        for est_name, estimator in named_estimators:
            cv_results = cross_validate(
                estimator,
                X,
                y,
                cv=cv,
                scoring=self.scorers_,
                return_train_score=True,
            )

            # average across folds
            est_scores = {}
            for metric in self.scorers_.keys():
                est_scores[metric] = {
                    "train": float(np.mean(cv_results[f"train_{metric}"])),
                    "test": float(np.mean(cv_results[f"test_{metric}"])),
                }

            results[est_name] = est_scores

        self.results = self._to_df(results)
        return self.results
