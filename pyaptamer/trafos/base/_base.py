"""Base transformation class."""

__author__ = ["fkiraly", "siddharth7113"]

import pandas as pd
from skbase.base import BaseEstimator

from pyaptamer.data import MoleculeLoader


class BaseTransform(BaseEstimator):
    """Base class for all transformations.

    Most transformers work on one column, the sequence column. They say so
    with the tag ``capability:multivariate`` set to False, and their ``_fit``
    and ``_transform`` always receive a frame with exactly one column.

    The base class makes that work for wider tables too. When ``fit`` gets a
    frame with several columns, it looks for the columns holding the kind of
    data the transformer needs, given by the tag ``input_type`` (strings by
    default). If exactly one column matches, the transformer runs on that
    column and the other columns are returned unchanged. If none or several
    match, ``fit`` raises and points the user to
    ``pyaptamer.trafos.compose.ApplyToCols``, which takes the column name.

    To write a new transformer: set the tags, implement ``_transform`` for a
    one-column frame, and implement ``_fit`` if the transformer learns
    anything from the data. Read the column by position, ``X.iloc[:, 0]``,
    since its name is whatever the user's table calls it.
    """

    _tags = {
        "object_type": "transformer",
        "input_type": "string",
        "capability:y": False,
        "output_type": "numeric",
        "property:fit_is_empty": False,
        "property:elementwise": False,
        "capability:multivariate": False,
    }

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        """Fit the transformer to the data.

        Parameters
        ----------
        X : pd.DataFrame or MoleculeLoader, shape (n_samples, n_features)
            Input data to fit the transformer. A univariate transformer
            accepts a frame with several columns if exactly one of them holds
            the ``input_type`` of the transformer, strings by default. It is
            fitted on that column only.
        y : array-like, shape (n_samples,), optional
            Target values. Only used if the transformer has
            the tag ``capability:y`` set to True.

        Returns
        -------
        self : object
            Returns self.
        """
        X_inner, y_inner = self._check_X_y(X, y)
        self._target_column_ = self._select_column(X_inner)
        if self._target_column_ is not None:
            X_inner = X_inner[[self._target_column_]]

        if not self.get_tag("property:fit_is_empty", False):
            self._fit(X=X_inner, y=y_inner)

        self._is_fitted = True
        return self

    def _fit(self, X, y=None):
        """Fit the transformer to the data.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data to fit the transformer. If the tag
            ``capability:multivariate`` is False, X has exactly one column.
            Select it by position. Its name is not fixed.
        y : array-like, shape (n_samples,), optional
            Target values. Only used if the transformer has
            the tag ``capability:y`` set to True.

        Returns
        -------
        self : object
            Returns self.
        """
        raise ValueError(
            "abstract method _fit called, this should be implemented in the subclass"
        )

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame or MoleculeLoader, shape (n_samples, n_features)
            Input data to transform. A univariate transformer fitted on a
            frame with several columns transforms the column found in ``fit``
            and returns the other columns unchanged, joined on the row index.
            Transformed columns are named ``<column>__<name>`` if there are
            several, and keep the column name if there is one.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, n_features_transformed)
            Transformed data.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called before.
        ValueError
            If ``fit`` found a column and X does not have it, or ``fit`` saw a
            one-column frame and X has several columns.
        """
        self.check_is_fitted(method_name="transform")
        X_inner = self._check_X(X)
        column = self._target_column_

        if column is None:
            if self._select_column(X_inner) is not None:
                raise ValueError(
                    f"{type(self).__name__} was fitted on a one-column frame, but "
                    f"X has columns {list(X_inner.columns)}. Fit and transform "
                    "on the same columns."
                )
            return self._transform(X=X_inner)

        if column not in X_inner.columns:
            raise ValueError(
                f"{type(self).__name__} was fitted on column {column!r}, "
                f"but X has columns {list(X_inner.columns)}."
            )

        Xt = self._transform(X=X_inner[[column]])
        return self._splice_transformed(X_inner, column, Xt)

    def _transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data to transform. If the tag ``capability:multivariate``
            is False, X has exactly one column. Select it by position. Its
            name is not fixed.

        Returns
        -------
        X : array-like, shape (n_samples, n_features_transformed)
            Transformed data.
        """
        if self.get_tag("property:elementwise", False):
            return X.map(self._transform_element)

        raise ValueError(
            "abstract method _transform called, "
            "this should be implemented in the subclass"
        )

    def _transform_element(self, X):
        """Transform the data - for elementwise transformers.

        Called only if the tag ``"property:elementwise"`` is True.

        Parameters
        ----------
        X : entry of X passed to transform
            Input data to transform.

        Returns
        -------
        X : array-like, shape (n_samples, n_features_transformed)
            Transformed data.
        """
        raise ValueError(
            "abstract method _transform_element called, "
            "since tag 'property:elementwise' is True, "
            "this should be implemented in the subclass"
        )

    def fit_transform(self, X, y=None):
        """Fit to data and transform the same data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to transform.
        y : array-like, shape (n_samples,), optional, default=None
            Target values. Only used if the transformer has
            the tag ``capability:y`` set to True.

        Returns
        -------
        X : array-like, shape (n_samples, n_features_transformed)
            Transformed data.
        """
        return self.fit(X, y).transform(X)

    def _check_X_y(self, X, y):  # noqa: N802
        """Check X and y inputs.

        Coerces X to a pd.DataFrame.
        """
        if isinstance(X, MoleculeLoader):
            X = X.to_dataframe()
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a MoleculeLoader instance"
                " or a pandas DataFrame. "
                f"Got {type(X)} instead."
            )

        return X, y

    def _check_X(self, X):  # noqa: N802
        """Check X input.

        Same as _check_X_y but only for X.
        """
        X, _ = self._check_X_y(X, None)
        return X

    def _select_column(self, X):
        """Find the column a one-column transformer should work on.

        Returns None when the transformer should see all of X: either it is
        multivariate, or X has a single column anyway. Otherwise returns the
        name of the one column whose values match the ``input_type`` tag.

        Raises
        ------
        ValueError
            If X has several columns and none, or more than one, of them
            match the ``input_type`` tag.
        """
        if self.get_tag("capability:multivariate", False) or X.shape[1] == 1:
            return None

        kind = self.get_tag("input_type")
        matches = [
            c
            for c in X.columns
            if pd.api.types.infer_dtype(X[c].dropna().head(), skipna=True) == kind
        ]

        if len(matches) == 1:
            return matches[0]

        name = type(self).__name__
        if not matches:
            raise ValueError(
                f"{name} works on one column of {kind}s, but X has no column "
                f"of {kind}s. X has columns {list(X.columns)}."
            )
        raise ValueError(
            f"{name} works on one column of {kind}s, but X has {len(matches)}: "
            f"{matches}. Pass one of them as cols to "
            f"pyaptamer.trafos.compose.ApplyToCols, for example "
            f"ApplyToCols({name}(...), cols={matches[0]!r})."
        )

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
        if Xt.shape[1] == 1:
            Xt = Xt.set_axis([column], axis=1)
        else:
            Xt = Xt.set_axis([f"{column}__{c}" for c in Xt.columns], axis=1)

        rest = X.drop(columns=column)

        clash = rest.columns.intersection(Xt.columns).tolist()
        if clash:
            raise ValueError(
                f"{type(self).__name__} would name its output columns "
                f"{clash}, but X already has columns with these names. Rename "
                "or drop them first."
            )

        if not Xt.index.equals(X.index):
            if not X.index.is_unique:
                raise ValueError(
                    f"{type(self).__name__} dropped rows, but the row index of "
                    "X is not unique, so the other columns cannot be lined up "
                    "with the result. Give X a unique index first."
                )
            rest = rest.loc[Xt.index]
        position = X.columns.get_loc(column)
        before = rest.iloc[:, :position]
        after = rest.iloc[:, position:]
        return pd.concat([before, Xt, after], axis=1)
