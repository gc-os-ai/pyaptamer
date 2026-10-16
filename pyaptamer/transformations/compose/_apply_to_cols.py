"""Apply a transformer to one named column of a frame."""

__author__ = ["siddharth7113"]
__all__ = ["ApplyToCols"]

import pandas as pd

from pyaptamer.transformations.base import BaseTransform


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

    Univariate transformers accept a one-column frame only. This wrapper
    applies one to a chosen column of a table with several columns.

    Parameters
    ----------
    transformer : BaseTransform
        The transformer to run. ``fit`` works on a copy, so the object you
        pass in is not changed.
    cols : str
        The name of the column to run the transformer on.

    Examples
    --------
    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.transformations.compose import ApplyToCols
    >>> from pyaptamer.transformations.encode import KMerFrequencies
    >>>
    >>> X = MoleculeLoader(
    ...     data={"id": ["r1", "r2"], "sequence": ["ACGT", "AAAA"], "round": [1, 2]}
    ... )
    >>> Xt = ApplyToCols(KMerFrequencies(k=1), cols="sequence").fit_transform(X)
    >>> Xt.columns.tolist()
    ['id', 'sequence__0', 'sequence__1', 'sequence__2', 'sequence__3', 'round']

    Two sequence columns take one wrapper each, for example in a Pipeline:

    >>> from sklearn.pipeline import Pipeline
    >>> from pyaptamer.transformations.encode import PSeAAC
    >>>
    >>> X = MoleculeLoader(
    ...     data={
    ...         "aptamer": ["ACGTACGT"],
    ...         "protein": ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"],
    ...         "round": [3],
    ...     }
    ... )
    >>> steps = Pipeline(
    ...     [
    ...         ("aptamer", ApplyToCols(KMerFrequencies(k=1), cols="aptamer")),
    ...         ("protein", ApplyToCols(PSeAAC(), cols="protein")),
    ...     ]
    ... )
    >>> steps.fit_transform(X).shape
    (1, 355)
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

    def _splice_transformed(self, X, column, Xt):
        """Put the transformed output back into the table.

        X is the full input table, ``column`` the name of the column that was
        transformed, and Xt the transformer's output for that column. The
        result is X with ``column`` replaced by Xt, in the same position.

        Rows are matched by index. If the transformer removed rows, so that
        Xt has fewer rows than X, those rows are removed from the other
        columns too.

        If Xt has several columns they are renamed ``<column>__<name>``, for
        example ``sequence__0``. If Xt has one column it keeps the name
        ``column``.

        Raises
        ------
        ValueError
            If a renamed output column already exists in X, or if rows were
            dropped and the index of X is not unique, so the other columns
            cannot be lined up.
        """
        name = type(self.transformer_).__name__
        if Xt.shape[1] == 1:
            Xt = Xt.set_axis([column], axis=1)
        else:
            Xt = Xt.set_axis([f"{column}__{c}" for c in Xt.columns], axis=1)

        rest = X.drop(columns=column)

        clash = rest.columns.intersection(Xt.columns).tolist()
        if clash:
            raise ValueError(
                f"{name} would name its output columns {clash}, but X already "
                "has columns with these names."
            )

        if not Xt.index.equals(X.index):
            if not X.index.is_unique:
                raise ValueError(
                    f"{name} dropped rows, but the row index of X is not "
                    "unique, so the other columns cannot be lined up with the "
                    "result."
                )
            rest = rest.loc[Xt.index]
        position = X.columns.get_loc(column)
        before = rest.iloc[:, :position]
        after = rest.iloc[:, position:]
        return pd.concat([before, Xt, after], axis=1)

    @classmethod
    def get_test_params(cls):
        """Parameter sets for the shared transformer tests."""
        from pyaptamer.transformations.encode import GreedyEncoder, KMerFrequencies

        words = {"A": 1, "C": 2, "G": 3, "T": 4}
        return [
            {"transformer": GreedyEncoder(words=words), "cols": "aptamer"},
            {"transformer": KMerFrequencies(k=1), "cols": "aptamer"},
        ]
