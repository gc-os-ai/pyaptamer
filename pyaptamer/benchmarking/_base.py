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
        Summary DataFrame produced by :meth:`run`.

        - Index: pandas.MultiIndex with two levels (names shown in parentheses)
            - level 0 "estimator": estimator name
            - level 1 "metric": evaluator name
        - Columns: ["train", "test"] (both floats)
        - Cell values: mean scores (float) computed across CV folds:
            - "train" = mean of cross_validate(...)[f"train_{metric}"]
            - "test"  = mean of cross_validate(...)[f"test_{metric}"]

    raw_results_ : pd.DataFrame or None
        Per-fold scores produced by :meth:`run` when ``return_raw=True``.

        - Index: pandas.MultiIndex with three levels
            - level 0 "estimator": estimator name
            - level 1 "metric": evaluator name
            - level 2 "fold": fold index (0-based)
        - Columns: ["train", "test"] (both floats)
        - Cell values: raw per-fold scores, directly compatible with
          ``sktime``'s ``Evaluator`` for Friedman tests and Critical
          Difference diagrams.

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

    def __init__(self, estimators, metrics, X, y, cv=None, labels=None):
        self.estimators = estimators if isinstance(estimators, list) else [estimators]
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.X = X
        self.y = y
        self.cv = cv
        self.labels = labels
        self.results = None
        self.raw_results_ = None

    def _to_scorers(self, metrics):
        """Convert metric callables to a dict of scorers."""
        scorers = {}
        for metric in metrics:
            if not callable(metric):
                raise ValueError("Each metric should be a callable.")
            name = (
                metric.__name__
                if hasattr(metric, "__name__")
                else metric.__class__.__name__
            )
            scorers[name] = make_scorer(metric)
        return scorers

    def _to_df(self, results):
        """Convert nested mean results to a summary DataFrame."""
        records = []
        index = []

        for est_name, est_scores in results.items():
            for metric_name, scores in est_scores.items():
                records.append(scores)
                index.append((est_name, metric_name))

        index = pd.MultiIndex.from_tuples(index, names=["estimator", "metric"])
        return pd.DataFrame(records, index=index, columns=["train", "test"])

    def _to_raw_df(self, raw_results):
        """Convert nested per-fold results to a raw DataFrame.

        The resulting DataFrame is directly compatible with ``sktime``'s
        ``Evaluator`` class for Friedman tests and Critical Difference diagrams.
        """
        records = []
        index = []

        for est_name, est_scores in raw_results.items():
            for metric_name, fold_scores in est_scores.items():
                for fold_idx, (train_score, test_score) in enumerate(
                    zip(fold_scores["train"], fold_scores["test"])
                ):
                    records.append({"train": train_score, "test": test_score})
                    index.append((est_name, metric_name, fold_idx))

        index = pd.MultiIndex.from_tuples(
            index, names=["estimator", "metric", "fold"]
        )
        return pd.DataFrame(records, index=index, columns=["train", "test"])

    def run(self, return_raw=False):
        """
        Train each estimator and evaluate with cross-validation.

        Parameters
        ----------
        return_raw : bool, default=False
            If ``False`` (default), returns only a summary DataFrame with
            mean scores across folds.

            If ``True``, returns a tuple ``(summary, raw)`` where ``raw`` is
            a per-fold DataFrame with a three-level MultiIndex
            ``(estimator, metric, fold)``. The ``raw`` DataFrame is directly
            compatible with ``sktime``'s ``Evaluator`` class for Friedman
            tests and Critical Difference diagrams.

        Returns
        -------
        results : pd.DataFrame
            Summary DataFrame with mean scores.

            - Index: pandas.MultiIndex ``(estimator, metric)``
            - Columns: ["train", "test"] (floats)

        (results, raw_results) : tuple[pd.DataFrame, pd.DataFrame]
            Returned only when ``return_raw=True``. ``raw_results`` has a
            three-level MultiIndex ``(estimator, metric, fold)`` and contains
            the raw per-fold scores.
        """
        self.scorers_ = self._to_scorers(self.metrics)
        results = {}
        raw_results = {}

        if self.labels is not None:
            if len(self.labels) != len(self.estimators):
                raise ValueError("Length of labels must match length of estimators.")
            names = self.labels
        else:
            counts = {}
            for est in self.estimators:
                name = est.__class__.__name__
                counts[name] = counts.get(name, 0) + 1

            names = []
            seen = {}
            for est in self.estimators:
                name = est.__class__.__name__
                if counts[name] > 1:
                    seen[name] = seen.get(name, 0) + 1
                    names.append(f"{name}_{seen[name]}")
                else:
                    names.append(name)

        for estimator, est_name in zip(self.estimators, names):

            cv_results = cross_validate(
                estimator,
                self.X,
                self.y,
                cv=self.cv,
                scoring=self.scorers_,
                return_train_score=True,
            )

            # mean scores across folds (summary)
            est_scores = {}
            # raw per-fold scores (for sktime Evaluator compatibility)
            est_raw_scores = {}
            for metric in self.scorers_.keys():
                train_folds = cv_results[f"train_{metric}"]
                test_folds = cv_results[f"test_{metric}"]
                est_scores[metric] = {
                    "train": float(np.mean(train_folds)),
                    "test": float(np.mean(test_folds)),
                }
                est_raw_scores[metric] = {
                    "train": train_folds.tolist(),
                    "test": test_folds.tolist(),
                }

            results[est_name] = est_scores
            raw_results[est_name] = est_raw_scores

        self.results = self._to_df(results)
        self.raw_results_ = self._to_raw_df(raw_results)

        if return_raw:
            return self.results, self.raw_results_
        return self.results
