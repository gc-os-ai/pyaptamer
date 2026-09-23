"""Apply a transformer to one named column of a frame."""

__author__ = ["siddharth7113"]
__all__ = ["ApplyToCols"]

from pyaptamer.trafos.base import BaseTransform


class ApplyToCols(BaseTransform):
    """Run a transformer on one column of a table and keep the other columns.

    ``cols`` is the name of the column to work on, for example ``"sequence"``.
    The transformer is fitted on that column and its output replaces that
    column in the table. Every other column is returned as it was.

    Rows are matched by their index. If the transformer removes rows, the
    same rows are removed from the other columns too, so the table stays
    aligned.

    Output column names: if the transformer returns several columns they are
    called ``<cols>__<name>``, for example ``sequence__0``. If it returns one
    column, that column keeps the name in ``cols``.

    When to use this: a transformer that works on one column finds that
    column on its own when it is the only column of strings in the table.
    If the table has more than one string column, say an ``id`` column next
    to ``sequence``, it cannot tell which one to use. This wrapper tells it.

    Parameters
    ----------
    transformer : BaseTransform
        The transformer to run. ``fit`` works on a copy, so the object you
        pass in is not changed.
    cols : str
        The name of the column to run the transformer on.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.trafos.compose import ApplyToCols
    >>> from pyaptamer.trafos.encode import KMerFrequencies
    >>>
    >>> X = pd.DataFrame({"id": ["r1", "r2"], "sequence": ["ACGT", "AAAA"]})
    >>> Xt = ApplyToCols(KMerFrequencies(k=1), cols="sequence").fit_transform(X)
    >>> Xt.columns.tolist()
    ['id', 'sequence__0', 'sequence__1', 'sequence__2', 'sequence__3']
    """

    _tags = {
        "authors": ["siddharth7113"],
        "maintainers": ["siddharth7113"],
        "capability:multivariate": True,
        "property:fit_is_empty": False,
    }

    def __init__(self, transformer, cols):
        self.transformer = transformer
        self.cols = cols
        super().__init__()

    def _check_cols(self, X):
        """Raise if ``cols`` is not a single column name present in X."""
        if not isinstance(self.cols, str):
            raise TypeError(
                f"cols must be one column name as a str, got {self.cols!r}."
            )
        if self.cols not in X.columns:
            raise ValueError(
                f"{self.cols!r} is not a column of X. X has columns {list(X.columns)}."
            )

    def _fit(self, X, y=None):
        self._check_cols(X)
        self.transformer_ = self.transformer.clone()
        self.transformer_.fit(X[[self.cols]], y)
        return self

    def _transform(self, X):
        self._check_cols(X)
        Xt = self.transformer_.transform(X[[self.cols]])
        return self._splice_transformed(X, self.cols, Xt)

    @classmethod
    def get_test_params(cls):
        """Parameter sets for the shared transformer tests."""
        from pyaptamer.trafos.encode import GreedyEncoder, KMerFrequencies

        words = {"A": 1, "C": 2, "G": 3, "T": 4}
        return [
            {"transformer": GreedyEncoder(words=words), "cols": "aptamer"},
            {"transformer": KMerFrequencies(k=1), "cols": "aptamer"},
        ]
