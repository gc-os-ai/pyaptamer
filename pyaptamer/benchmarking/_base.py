__author__ = "satvshr"
__all__ = ["Benchmarking"]

import copy

import numpy as np
import pandas as pd
from skbase.base import BaseObject
from sklearn.model_selection import cross_validate, train_test_split

from pyaptamer.utils.tag_checks import task_check


class Benchmarking(BaseObject):
    """
    Benchmark estimators on train/test splits or cross-validation.

    You can either:

    - pass `X, y` (feature matrix and labels/targets) and let this class
        split into train/test automatically (if `cv=None`); or
    - pass explicit `train_X, train_y, test_X, test_y` (if `cv=None`); or
    - pass `X, y` along with `cv` to use cross-validation.

    Parameters
    ----------
    estimators : list[estimator] | estimator
        List of sklearn-like estimators implementing `fit` and `predict`.
    evaluators : list[callable] | callable
        List of callables with signature ``(y_true, y_pred) -> float``.
    X : array-like, optional
        Feature matrix. Used together with `y` if explicit train/test splits
        are not provided.
    y : array-like, optional
        Target vector. Used together with `X` if explicit train/test splits
        are not provided.
    train_X : array-like, optional
        Training feature matrix (ignored if `cv` is given).
    train_y : array-like, optional
        Training labels/targets (ignored if `cv` is given).
    test_X : array-like, optional
        Test feature matrix (ignored if `cv` is given).
    test_y : array-like, optional
        Test labels/targets (ignored if `cv` is given).
    test_size : float, default=0.2
        Fraction of data to reserve for the test split when splitting `X, y`.
        Ignored if `cv` is provided.
    stratify : bool, default=True
        If True, and the task is classification, stratify the train/test split
        using `y`. Ignored if `cv` is provided.
    random_state : int or None, default=42
        Random state for reproducibility in splits.
    cv : int, CV splitter, or None, default=None
        Cross-validation strategy. If provided, results are averaged
        across folds and returned in the same format as hold-out mode.

    Attributes
    ----------
    results : pd.DataFrame
        Results table after calling :meth:`run`.

        - Rows = MultiIndex (estimator, metric)
        - Cols = ["train", "test"]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> from pyaptamer.benchmarking._base import Benchmarking
    >>> from pyaptamer.aptanet import AptaNetPipeline
    >>> from pyaptamer.datasets import load_csv_dataset
    >>> X, y = load_csv_dataset("train_li2014", "label", return_X_y=True)
    >>> X = X[:10]  # smaller example
    >>> y = y[:10]
    >>> y = np.where(y == "positive", 1, 0)
    >>> clf = AptaNetPipeline()
    >>> bench = Benchmarking(
    ...     estimators=[clf],
    ...     evaluators=[accuracy_score],
    ...     X=X,
    ...     y=y,
    ... )
    >>> summary = bench.run()  # doctest: +SKIP
    """

    _tags = {"tasks": ["classification", "regression"]}

    def __init__(
        self,
        estimators,
        evaluators,
        X=None,
        y=None,
        train_X=None,
        train_y=None,
        test_X=None,
        test_y=None,
        test_size=0.2,
        stratify=True,
        random_state=None,
        cv=None,
    ):
        self.estimators = estimators if isinstance(estimators, list) else [estimators]
        self.evaluators = evaluators if isinstance(evaluators, list) else [evaluators]
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        self.cv = cv

        # case 1: explicit train/test provided
        if (
            train_X is not None
            and train_y is not None
            and test_X is not None
            and test_y is not None
        ):
            if self.cv is not None:
                raise ValueError(
                    "Cannot use both explicit train/test splits and cross-validation. "
                    "Either provide (train_X, train_y, test_X, test_y) or (X, y, cv)."
                )
            self.train_X, self.train_y = train_X, train_y
            self.test_X, self.test_y = test_X, test_y

        # case 2: (X, y) with cv
        elif self.cv is not None:
            if X is None or y is None:
                raise ValueError("Provide (X, y) when using cross-validation.")
            self.X, self.y = X, y

        # case 3: (X, y) with hold-out split
        elif X is not None and y is not None:
            (
                self.train_X,
                self.test_X,
                self.train_y,
                self.test_y,
            ) = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if self.stratify else None,
                shuffle=True,
            )

        else:
            raise ValueError(
                "Provide either (X, y), (X, y, cv), or"
                "(train_X, train_y, test_X, test_y)."
            )

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
        Train each estimator and evaluate with hold-out or cross-validation.

        Returns
        -------
        pd.DataFrame
            Results table with rows = (estimator, metric),
            cols = ["train", "test"].
        """
        task_check(self)
        results = {}

        if hasattr(self, "X") and hasattr(self, "y"):
            for estimator in self.estimators:
                est_name = estimator.__class__.__name__

                scoring = {
                    getattr(
                        evaluator,
                        "__name__",
                        getattr(evaluator, "name", evaluator.__class__.__name__),
                    ): evaluator
                    for evaluator in self.evaluators
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

        else:
            for estimator in self.estimators:
                est_name = estimator.__class__.__name__
                model = copy.deepcopy(estimator)
                model.fit(self.train_X, self.train_y)

                est_scores = {}
                for split_name, (X_split, y_split) in {
                    "train": (self.train_X, self.train_y),
                    "test": (self.test_X, self.test_y),
                }.items():
                    y_pred = model.predict(X_split)
                    for evaluator in self.evaluators:
                        eval_name = getattr(
                            evaluator,
                            "__name__",
                            getattr(evaluator, "name", evaluator.__class__.__name__),
                        )
                        est_scores.setdefault(eval_name, {})[split_name] = evaluator(
                            y_split, y_pred
                        )

                results[est_name] = est_scores

        self.results = self._to_df(results)
        return self.results
