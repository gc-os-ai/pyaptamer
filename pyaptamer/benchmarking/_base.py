from pyaptamer.utils.tag_checks import task_check


class Benchmark:
    """Base class for benchmarking different estimators on various datasets."""

    _tags = {"tasks": ["classification", "regression"]}

    def __init__(self, estimators, dataset_Loaders, evaluators, task):
        self.estimators = [estimators]
        self.dataset_Loaders = [dataset_Loaders]
        self.evaluators = [evaluators]
        self.task = task
        task_check(self)

    def run(self):
        results = {}
        for _estimator in self.estimators:
            for dataset_Loader in self.dataset_Loaders:
                dataset_name = dataset_Loader.id
                results[dataset_name] = {}
