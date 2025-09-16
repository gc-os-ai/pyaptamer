__author__ = "satvshr"
__all__ = ["Benchmarking"]

import numpy as np
import pandas as pd
from skbase.base import BaseObject
from sklearn.model_selection import check_cv, train_test_split

from pyaptamer.utils.tag_checks import task_check


class Benchmarking(BaseObject):
    """
    Benchmark estimators on train/test splits or cross-validation.

    You can either:

      - pass `X, y` (feature matrix and labels/targets) and let this class
        split into train/test for you (if cv=None); or
      - pass explicit `train_X, train_y, test_X, test_y` (if cv=None); or
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
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=0)
    >>> bench = Benchmarking(
    ...     estimators=[LogisticRegression(max_iter=1000)],
    ...     evaluators=[accuracy_score],
    ...     X=X,
    ...     y=y,
    ...     test_size=0.3,
    ...     random_state=42,
    ... )
    >>> results = bench.run()
    >>> print(results)
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
        random_state=42,
        cv=None,
    ):
        self.estimators = estimators if isinstance(estimators, list) else [estimators]
        self.evaluators = evaluators if isinstance(evaluators, list) else [evaluators]
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        self.cv = cv

        if self.cv is None:
            if X is not None and y is not None:
                (
                    self.train_X,
                    self.test_X,
                    self.train_y,
                    self.test_y,
                ) = self._split_dataset(X, y)
            elif (
                train_X is not None
                and train_y is not None
                and test_X is not None
                and test_y is not None
            ):
                self.train_X, self.train_y = train_X, train_y
                self.test_X, self.test_y = test_X, test_y
            else:
                raise ValueError(
                    "Provide either (X, y) or (train_X, train_y, test_X, test_y)"
                    "when cv=None."
                )
        else:
            if X is None or y is None:
                raise ValueError("Provide (X, y) when using cross-validation.")
            self.X, self.y = X, y

        self.results = None

    def _split_dataset(self, X, y):
        """Split into train/test arrays."""
        stratify_vec = y if self.stratify else None
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_vec,
            shuffle=True,
        )

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
        import copy

        from sklearn.base import clone

        task_check(self)

        results = {}

        if self.cv is not None:
            # Cross-validation mode
            cv = check_cv(self.cv, y=self.y, classifier=None)

            for estimator in self.estimators:
                est_name = estimator.__class__.__name__
                metric_scores = {
                    metric.__name__: {"train": [], "test": []}
                    for metric in self.evaluators
                }

                for train_idx, test_idx in cv.split(self.X, self.y):
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]

                    model = clone(estimator)
                    model.fit(X_train, y_train)

                    # evaluate on both train and test fold parts
                    for split_name, (X_split, y_split) in {
                        "train": (X_train, y_train),
                        "test": (X_test, y_test),
                    }.items():
                        y_pred = model.predict(X_split)
                        for evaluator in self.evaluators:
                            eval_name = getattr(
                                evaluator,
                                "__name__",
                                getattr(
                                    evaluator, "name", evaluator.__class__.__name__
                                ),
                            )
                            metric_scores[eval_name][split_name].append(
                                evaluator(y_split, y_pred)
                            )

                # average across folds
                results[est_name] = {
                    metric: {
                        "train": float(np.mean(scores["train"])),
                        "test": float(np.mean(scores["test"])),
                    }
                    for metric, scores in metric_scores.items()
                }

        else:
            # Hold-out mode
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
                    scores = {}
                    for evaluator in self.evaluators:
                        eval_name = getattr(
                            evaluator,
                            "__name__",
                            getattr(evaluator, "name", evaluator.__class__.__name__),
                        )
                        scores[eval_name] = evaluator(y_split, y_pred)
                    for eval_name, score in scores.items():
                        est_scores.setdefault(eval_name, {})[split_name] = score

                results[est_name] = est_scores

        self.results = self._to_df(results)
        return self.results
