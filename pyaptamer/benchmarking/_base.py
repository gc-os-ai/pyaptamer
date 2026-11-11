__author__ = "satvshr"
__all__ = ["Benchmarking"]

import warnings

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
    metrics : list[callable | dict] | callable | dict
        Metrics to evaluate. Supported forms:
        - callable with signature `(y_true, y_pred) -> float` (legacy behavior)
        - dict with keys:
            - `metric` (required): callable scoring function
            - `name` (optional): custom metric name in output index
            - `response_method` (optional): one of
              {"predict", "predict_proba", "decision_function"}
            - `greater_is_better` (optional): bool forwarded to `make_scorer`
            - `kwargs` (optional): dict of extra kwargs for metric callable
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

    def __init__(self, estimators, metrics, X, y, cv=None):
        self.estimators = estimators if isinstance(estimators, list) else [estimators]
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.X = X
        self.y = y
        self.cv = cv
        self.results = None

    @staticmethod
    def _metric_name(metric):
        """Return a stable display name for a metric callable."""
        return (
            metric.__name__
            if hasattr(metric, "__name__")
            else metric.__class__.__name__
        )

    def _normalize_metric_spec(self, metric_spec):
        """Normalize metric spec into scorer-compatible metadata."""
        if callable(metric_spec):
            return {
                "name": self._metric_name(metric_spec),
                "metric": metric_spec,
                "response_method": None,
                "greater_is_better": None,
                "kwargs": {},
            }

        if not isinstance(metric_spec, dict):
            raise ValueError("Each metric should be a callable or a dict metric spec.")

        metric = metric_spec.get("metric")
        if not callable(metric):
            raise ValueError("Metric spec requires a callable under key 'metric'.")

        response_method = metric_spec.get("response_method")
        if response_method is not None and response_method not in {
            "predict",
            "predict_proba",
            "decision_function",
        }:
            raise ValueError(
                "Metric spec key 'response_method' must be one of "
                "{'predict', 'predict_proba', 'decision_function'}."
            )

        kwargs = metric_spec.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise ValueError("Metric spec key 'kwargs' must be a dictionary.")

        greater_is_better = metric_spec.get("greater_is_better")
        if greater_is_better is not None and not isinstance(greater_is_better, bool):
            raise ValueError("Metric spec key 'greater_is_better' must be boolean.")

        return {
            "name": metric_spec.get("name", self._metric_name(metric)),
            "metric": metric,
            "response_method": response_method,
            "greater_is_better": greater_is_better,
            "kwargs": kwargs,
        }

    def _build_scorer(
        self, metric, response_method=None, greater_is_better=None, kwargs=None
    ):
        """Build sklearn scorer with version-compatible response handling."""
        kwargs = kwargs or {}
        scorer_kwargs = dict(kwargs)
        if greater_is_better is not None:
            scorer_kwargs["greater_is_better"] = greater_is_better

        if response_method is None or response_method == "predict":
            return make_scorer(metric, **scorer_kwargs)

        try:
            return make_scorer(metric, response_method=response_method, **scorer_kwargs)
        except TypeError:
            # sklearn<1.4 compatibility path
            if response_method == "predict_proba":
                return make_scorer(metric, needs_proba=True, **scorer_kwargs)
            if response_method == "decision_function":
                return make_scorer(metric, needs_threshold=True, **scorer_kwargs)
            raise

    def _to_scorers(self, metrics):
        """Convert metric callables to a dict of scorers."""
        scorers = {}
        known_score_metrics = {
            "roc_auc_score",
            "average_precision_score",
            "log_loss",
            "brier_score_loss",
        }

        for metric_spec in metrics:
            spec = self._normalize_metric_spec(metric_spec)

            if (
                spec["response_method"] is None
                and self._metric_name(spec["metric"]) in known_score_metrics
            ):
                warnings.warn(
                    f"Metric '{spec['name']}' typically expects continuous scores. "
                    "Consider setting response_method='predict_proba' or "
                    "'decision_function' in the metric spec.",
                    stacklevel=2,
                )

            scorers[spec["name"]] = self._build_scorer(
                spec["metric"],
                response_method=spec["response_method"],
                greater_is_better=spec["greater_is_better"],
                kwargs=spec["kwargs"],
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

    def run(self):
        """
        Train each estimator and evaluate with cross-validation.

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

        for estimator in self.estimators:
            est_name = estimator.__class__.__name__

            cv_results = cross_validate(
                estimator,
                self.X,
                self.y,
                cv=self.cv,
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
