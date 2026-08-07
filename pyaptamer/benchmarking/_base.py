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
      to the `run()` method to use any cross-validation strategy;
    - if you want a fixed train/test split, pass a `PredefinedSplit`
      object as `cv`.

    Parameters
    ----------
    estimators : list[estimator] | estimator
        List of sklearn-like estimators implementing `fit` and `predict`.
        Can also be a list of tuples (name, estimator) for custom naming.
    metrics : list[callable | str] | callable | str
        List of callables with signature `(y_true, y_pred) -> float`,
        or string names compatible with sklearn (e.g., "accuracy", "f1").

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
    ... )
    >>> summary = bench.run(X=X, y=y, cv=cv)  # doctest: +SKIP
    """

    def __init__(self, estimators, metrics):
        # Handle estimators: can be list of estimators or list of (name, estimator) tuples
        if not isinstance(estimators, list):
            estimators = [estimators]
        
        self.estimators = []
        self.estimator_names = []
        
        for est in estimators:
            if isinstance(est, tuple) and len(est) == 2:
                # (name, estimator) tuple
                name, estimator = est
                self.estimator_names.append(name)
                self.estimators.append(estimator)
            else:
                # Just an estimator
                self.estimators.append(est)
                self.estimator_names.append(None)  # Will be auto-generated
        
        # Handle metrics: can be callables or strings
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.results = None

    def _to_scorers(self, metrics):
        """Convert metric callables or strings to a dict of scorers."""
        from sklearn.metrics import get_scorer
        
        scorers = {}
        for metric in metrics:
            if isinstance(metric, str):
                # String-based metric name (e.g., "accuracy", "f1")
                scorers[metric] = metric
            elif callable(metric):
                # Callable metric
                name = (
                    metric.__name__
                    if hasattr(metric, "__name__")
                    else metric.__class__.__name__
                )
                scorers[name] = make_scorer(metric)
            else:
                raise ValueError(
                    "Each metric should be a callable or a string metric name."
                )
        return scorers

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
            If you want to use an explicit train/test split, pass a
            `PredefinedSplit` object.

        Returns
        -------
        results : pd.DataFrame

            - Index: pandas.MultiIndex with two levels (names shown in parentheses)
                - level 0 "estimator": estimator name
                - level 1 "metric": evaluator name
            - Columns: ["train", "test"] (both floats)
            - Cell values: mean scores (float) computed across CV folds:
                - "train" = mean of cross_validate(...)[f"train_{metric}"]
                - "test"  = mean of cross_validate(...)[f"test_{metric}"]

        """
        self.scorers_ = self._to_scorers(self.metrics)
        results = {}
        
        # Generate unique estimator names, handling collisions
        used_names = {}
        final_names = []
        
        for i, (estimator, custom_name) in enumerate(zip(self.estimators, self.estimator_names)):
            if custom_name is not None:
                # Use custom name provided by user
                est_name = custom_name
            else:
                # Auto-generate name from class
                est_name = estimator.__class__.__name__
            
            # Handle name collisions by appending a counter
            if est_name in used_names:
                used_names[est_name] += 1
                est_name = f"{est_name}_{used_names[est_name]}"
            else:
                used_names[est_name] = 0
            
            final_names.append(est_name)

        # Run cross-validation for each estimator
        for estimator, est_name in zip(self.estimators, final_names):
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
