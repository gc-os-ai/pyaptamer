"""Add all `_tag` checking functions here."""


def task_check(estimator):
    tags = estimator.get_tags()
    if estimator.task not in tags["tasks"]:
        raise ValueError(
            f"Invalid task '{estimator.task}'. Must be one of {tags['tasks']}."
        )
