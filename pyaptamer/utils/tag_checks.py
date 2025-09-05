"""Add all `_tag` checking functions here."""


def task_check(estimator):
    if estimator.task not in estimator._tags["tasks"]:
        raise ValueError(
            f"Invalid task '{estimator.task}'. Must be a {estimator._tags['tasks']}."
        )
