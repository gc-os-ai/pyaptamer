__author__ = "satvshr"
__all__ = ["task_check"]

"""Add all `_tag` checking functions here."""


def task_check(estimator):
    """
    Validate that an estimator's `task` is declared in its scikit-learn tags.

    Parameters
    ----------
    estimator : object
        An estimator object that implements `get_tags()` and exposes a
        `.task` attribute indicating the intended task (e.g., "classification",
        "regression").

    Raises
    ------
    ValueError
        If `estimator.task` is not contained in the estimator's `"tasks"` tag.
    """
    tags = estimator.get_tags()
    if estimator.task not in tags["tasks"]:
        raise ValueError(
            f"Invalid task '{estimator.task}'. Must be one of {tags['tasks']}."
        )
