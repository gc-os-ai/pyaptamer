"""Base transformation class."""

__author__ = ["fkiraly", "siddharth7113"]
__all__ = ["BaseTransform"]

import pandas as pd
from skbase.base import BaseEstimator

from pyaptamer.data import MoleculeLoader


class BaseTransform(BaseEstimator):
    """Base class for all transformations.

    Most transformers work on one column, the sequence column. They say so
    with the tag ``capability:multivariate`` set to False, and their ``_fit``
    and ``_transform`` always receive a frame with exactly one column. Given
    a frame with several columns, ``fit`` and ``transform`` raise and point
    to ``pyaptamer.trafos.compose.ApplyToCols``, which runs the transformer
    on a named column and returns the other columns unchanged.

    To write a new transformer: set the tags, implement ``_transform`` for a
    one-column frame, and implement ``_fit`` if the transformer learns
    anything from the data. Read the column by position, ``X.iloc[:, 0]``,
    since its name is whatever the user's table calls it.
    """

    _tags = {
        "object_type": "transformer",
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
            accepts one column; for a table with several columns use
            ``pyaptamer.trafos.compose.ApplyToCols``.
        y : array-like, shape (n_samples,), optional
            Target values. Only used if the transformer has
            the tag ``capability:y`` set to True.

        Returns
        -------
        self : object
            Returns self.
        """
        X_inner, y_inner = self._check_X_y(X, y)

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
        raise NotImplementedError(
            "abstract method _fit called, this should be implemented in the subclass"
        )

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame or MoleculeLoader, shape (n_samples, n_features)
            Input data to transform. A univariate transformer accepts one
            column; for a table with several columns use
            ``pyaptamer.trafos.compose.ApplyToCols``.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, n_features_transformed)
            Transformed data.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called before.
        ValueError
            If the transformer is univariate and X has several columns.
        """
        self.check_is_fitted(method_name="transform")
        X_inner = self._check_X(X)
        Xt = self._transform(X=X_inner)
        return Xt

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

        raise NotImplementedError(
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
        raise NotImplementedError(
            "abstract method _transform_element called, "
            "since tag 'property:elementwise' is True, "
            "this should be implemented in the subclass"
        )

    def fit_transform(self, X, y=None):
        """Fit to data and transform the same data.

        Parameters
        ----------
        X : pd.DataFrame or MoleculeLoader, shape (n_samples, n_features)
            Input data to transform.
        y : array-like, shape (n_samples,), optional, default=None
            Target values. Only used if the transformer has
            the tag ``capability:y`` set to True.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, n_features_transformed)
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

        if not self.get_tag("capability:multivariate", False) and X.shape[1] > 1:
            raise ValueError(
                f"{type(self).__name__} works on a single column, but X has "
                f"columns {list(X.columns)}. Pass the column name as cols to "
                "pyaptamer.trafos.compose.ApplyToCols(transformer, cols)."
            )

        return X, y

    def _check_X(self, X):  # noqa: N802
        """Check X input.

        Same as _check_X_y but only for X.
        """
        X, _ = self._check_X_y(X, None)
        return X
