__author__ = "satvshr"
__all__ = ["Benchmarking"]

import numpy as np
import pandas as pd
from skbase.base import BaseObject
from sklearn.model_selection import train_test_split

from pyaptamer.utils.tag_checks import task_check


class Benchmarking(BaseObject):
    """
    Train on each train dataset and evaluate on every test dataset
    with each estimator & evaluator.

    You can either:

      - pass `datasets` (raw frames with columns ["aptamer","protein","y"] OR
          preprocessed ["X","y"]) and let this class preprocess + split for you; or
      - pass `train_datasets` and `test_datasets`
          which may be EITHER raw (["aptamer","protein","y"]) OR preprocessed
          (["X","y"]).

    To input datasets, you should input a dictionary of DataFrames, with the format
    being { "dataset_name": pd.DataFrame(...) }.

    Parameters
    ----------
    estimators : list[estimator] | estimator
        List of sklearn-like estimators with fit/predict.
    evaluators : list[callable] | callable
        List of callables with signature (y_true, y_pred) -> float.
    task : str
        "classification" or "regression".
    preprocessor : BasePreprocessor | None
        Used to convert raw ["aptamer","protein","y"] -> ["X","y"] when needed.
    datasets : dict[str, pd.DataFrame] | None
        Raw datasets to be split into train/test (if provided).
    train_datasets : dict[str, pd.DataFrame] | None
        Training datasets (raw or preprocessed).
    test_datasets : dict[str, pd.DataFrame] | None
        Test datasets (raw or preprocessed).
    test_size : float
        Fraction for the test split when using `datasets`.
    stratify : bool
        If True and task=="classification", use y for stratification
        (only when splitting).
    random_state : int | None
        Random state for reproducibility in splits.

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.metrics import accuracy_score
    >>> from pyaptamer.benchmarking._base import Benchmarking
    >>> from pyaptamer.aptanet import AptaNetPipeline
    >>> from pyaptamer.benchmarking.preprocessors.aptanet import AptaNetPreprocessor
    >>> from pyaptamer.datasets import load_csv_dataset
    >>> df = load_csv_dataset(
    ...     "train_li2014"
    ... )  # expects columns ['aptamer','protein','label']
    >>> df = df[:10]  # smaller example
    >>> df = df.rename(columns={"label": "y"})
    >>> df["y"] = df["y"].map({"negative": 0, "positive": 1})
    >>> pre = AptaNetPreprocessor()
    >>> clf = AptaNetPipeline()
    >>> bench = Benchmarking(
    ...     estimators=[clf],
    ...     evaluators=[accuracy_score],
    ...     task="classification",
    ...     preprocessor=pre,
    ...     datasets=df,
    ...     test_size=0.25,
    ...     stratify=True,
    ...     random_state=1337,
    ... )
    >>> bench.run()
    """

    _tags = {"tasks": ["classification", "regression"]}

    def __init__(
        self,
        estimators,
        evaluators,
        task,
        preprocessor=None,
        datasets=None,
        train_datasets=None,
        test_datasets=None,
        test_size=0.2,
        stratify=True,
        random_state=42,
    ):
        self.estimators = estimators if isinstance(estimators, list) else [estimators]
        self.evaluators = evaluators if isinstance(evaluators, list) else [evaluators]
        self.task = task
        self.preprocessor = preprocessor
        self.test_size = test_size
        self.stratify_flag = stratify
        self.random_state = random_state

        # accept either (datasets) or (train/test)
        # normalize datasets to dict
        self._raw_datasets = self._normalize_to_dict(datasets, prefix="dataset")
        self._train_datasets_in = self._normalize_to_dict(
            train_datasets, prefix="train"
        )
        self._test_datasets_in = self._normalize_to_dict(test_datasets, prefix="test")

        # prepared containers
        self.train_datasets = {}
        self.test_datasets = {}

        self.results = None

        # prepare datasets immediately
        self._build_datasets()

    def _build_datasets(self):
        """
        Build self.train_datasets and self.test_datasets from provided inputs.
        """
        # case A: user provided raw datasets to split
        if self._raw_datasets is not None:
            for name, df in self._raw_datasets.items():
                train, test, train_name, test_name = self._split_dataset(name, df)
                self.train_datasets[train_name] = train
                self.test_datasets[test_name] = test

        # case B: user provided train/test datasets (each may be raw or preprocessed)
        if self._train_datasets_in is not None:
            for name, df in self._train_datasets_in.items():
                self.train_datasets[name] = self._ensure_preprocessed(df)

        if self._test_datasets_in is not None:
            for name, df in self._test_datasets_in.items():
                self.test_datasets[name] = self._ensure_preprocessed(df)

        if not self.train_datasets or not self.test_datasets:
            raise ValueError(
                "No datasets prepared. Provide either `datasets` to split or "
                "`train_datasets` and `test_datasets` (raw or with ['X','y'])."
            )

        for name, df in {**self.train_datasets, **self.test_datasets}.items():
            self._validate_xy(df, name=name)

    def _normalize_to_dict(self, data, prefix):
        """Convert single DataFrame to dict or return existing dict."""
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return {f"{prefix}_0": data}
        if isinstance(data, dict):
            return data
        raise ValueError(f"Data must be DataFrame or dict, {type(data)} given.")

    def _ensure_preprocessed(self, df):
        """
        Ensure df has only ["X","y"].
        - If extra columns are present, warn and return only ["X","y"].
        - If missing, run the preprocessor and reduce to ["X","y"].
        """
        import warnings

        if {"X", "y"}.issubset(df.columns):
            if set(df.columns) != {"X", "y"}:
                warnings.warn(
                    f"Extra columns {set(df.columns) - {'X', 'y'}} found. "
                    "Only ['X','y'] will be kept.",
                    stacklevel=2,
                )
            return df[["X", "y"]]

        if self.preprocessor is None:
            raise ValueError(
                "Input DataFrame does not have ['X','y'] and no preprocessor"
                "was provided."
            )

        df_proc = self.preprocessor.transform(df)
        return df_proc[["X", "y"]]

    def _split_dataset(self, name, df):
        """
        Split a single dataset into train/test and return preprocessed splits.
        Works whether `df` is raw or already preprocessed.
        Returns (train_df, test_df, train_name, test_name).
        """
        df_proc = self._ensure_preprocessed(df)
        X, y = df_proc["X"], df_proc["y"]

        stratify_vec = (
            y if (self.task == "classification" and self.stratify_flag) else None
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_vec,
            shuffle=True,
        )

        train_df = pd.DataFrame({"X": X_train, "y": y_train}).reset_index(drop=True)
        test_df = pd.DataFrame({"X": X_test, "y": y_test}).reset_index(drop=True)

        train_name = f"{name}__train"
        test_name = f"{name}__test"
        return train_df, test_df, train_name, test_name

    def _validate_xy(self, df, name="dataset"):
        if not {"X", "y"}.issubset(df.columns):
            raise ValueError(f"{name} is missing required columns ['X','y'].")

        if len(df["X"]) != len(df["y"]):
            raise ValueError(f"{name} has mismatched X/y lengths.")

        if self.task == "classification":
            df["y"] = df["y"].astype(int)
            if pd.api.types.is_numeric_dtype(
                df["y"]
            ) and not pd.api.types.is_integer_dtype(df["y"]):
                raise ValueError(
                    f"{name}: classification target y must be categorical/integer, not"
                    "float."
                )
        elif self.task == "regression":
            if not pd.api.types.is_numeric_dtype(df["y"]):
                raise ValueError(f"{name}: regression target y must be numeric.")

        return df[["X", "y"]]

    def _to_df(self, results):
        """
        Convert nested dict results to summary statistics and print detailed DataFrames.

        Returns
        -------
        dict
            Summary statistics containing:
            - 'mean_scores': Average score per metric across all test sets
            - 'best_model': Name of estimator with highest average performance
            - 'best_score': Best score achieved
            - 'per_estimator_stats': Dictionary of statistics per estimator
        """
        dfs = {}
        summary_stats = {
            "mean_scores": {},
            "best_model": None,
            "best_score": 0,
            "per_estimator_stats": {},
        }

        for estimator, train_dict in results.items():
            train_df_blocks = []
            estimator_scores = []

            for train_set, test_dict in train_dict.items():
                df = pd.DataFrame(test_dict)  # metrics × test_sets
                df["train_set"] = train_set
                df = df.set_index(["train_set"], append=True)
                df = df.reorder_levels(["train_set", df.index.names[0]])
                train_df_blocks.append(df)

                # Collect scores for summary
                for metric in df.columns:
                    estimator_scores.extend(df[metric].values)

            dfs[estimator] = pd.concat(train_df_blocks)

            # Calculate per-estimator statistics
            mean_score = np.mean(estimator_scores)
            summary_stats["per_estimator_stats"][estimator] = {
                "mean_score": mean_score,
                "std_score": np.std(estimator_scores),
                "max_score": np.max(estimator_scores),
                "min_score": np.min(estimator_scores),
            }

            # Track best model
            if mean_score > summary_stats["best_score"]:
                summary_stats["best_score"] = mean_score
                summary_stats["best_model"] = estimator

        # Print detailed results
        for est, df in dfs.items():
            print(f"\n=== {est} ===")
            print(df)

        return summary_stats

    def run(self):
        """
        Train each estimator on each training dataset and evaluate on all test datasets.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping estimator_name -> DataFrame of metrics
            (rows = train_set, cols = test_sets).
        """
        import copy

        task_check(self)
        results = {}

        for estimator in self.estimators:
            estimator_name = estimator.__class__.__name__
            results[estimator_name] = {}

            for train_name, train_df in self.train_datasets.items():
                model = copy.deepcopy(estimator)
                model.fit(train_df["X"], train_df["y"])
                results[estimator_name][train_name] = {}

                for test_name, test_df in self.test_datasets.items():
                    y_true = test_df["y"]
                    y_pred = model.predict(test_df["X"])
                    test_results = {}

                    for evaluator in self.evaluators:
                        evaluator_name = getattr(
                            evaluator,
                            "__name__",
                            getattr(evaluator, "name", evaluator.__class__.__name__),
                        )
                        test_results[evaluator_name] = evaluator(y_true, y_pred)

                    results[estimator_name][train_name][test_name] = test_results

        self.results = self._to_df(results)
        return self.results
