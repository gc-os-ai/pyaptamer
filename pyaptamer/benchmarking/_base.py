from sklearn.base import clone


class Benchmark:
    """
    Train on each train dataset and evaluate on every test dataset
    with each estimator & evaluator.

    Input format:
      estimators: list[estimator]
      train_datasets: dict[str, pd.DataFrame] with columns ["X", "y"]
      test_datasets:  dict[str, pd.DataFrame] with columns ["X", "y"]
      evaluators: list[callable(y_true, y_pred) -> float]
    """

    def __init__(self, estimators, train_datasets, test_datasets, evaluators):
        self.estimators = estimators if isinstance(estimators, list) else [estimators]
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.evaluators = evaluators if isinstance(evaluators, list) else [evaluators]

    def run(self):
        # Each estimator is trained on each training dataset and evaluated on all test
        # datasets.
        # Results are nested as:
        # results[estimator_name][train_dataset_name][test_dataset_name][evaluator_name]
        # = score
        # Which looks like:
        # {
        #   "EstimatorA": {
        #     "train_dataset_1": {
        #       "test_dataset_1": {"evaluator_1": result, "evaluator_2": result},
        #       "test_dataset_2": {...}
        #     },
        #     "train_dataset_2": {...}
        #   },
        #   "EstimatorB": {
        #     ...
        #   }
        # }
        results = {}

        for estimator in self.estimators:
            estimator_name = estimator.__class__.__name__
            results[estimator_name] = {}

            for train_name, train_df in self.train_datasets.items():
                model = clone(estimator)
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

        return results
