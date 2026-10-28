"""Base transformation class."""

__author__ = ["fkiraly", "siddharth7113"]
__all__ = ["BaseTransform"]

from skbase.base import BaseEstimator
from sklearn.utils import InputTags, Tags, TargetTags, TransformerTags

from pyaptamer.utils._check_x import check_X


class BaseTransform(BaseEstimator):
    """Base class for all transformations.

    Most transformers work on one column, the sequence column. They say so
    with the tag ``capability:multivariate`` set to False, and their ``_fit``
    and ``_transform`` always receive a frame with exactly one column. Given
    a frame with several columns, ``fit`` and ``transform`` raise and point
    to ``pyaptamer.transformations.compose.ApplyToCols``, which runs the transformer
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
            ``pyaptamer.transformations.compose.ApplyToCols``.
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
            ``pyaptamer.transformations.compose.ApplyToCols``.

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

    def __sklearn_is_fitted__(self):
        """Return whether ``fit`` has been called.

        scikit-learn's ``check_is_fitted`` calls this, for example in a
        ``Pipeline`` whose last step is this transformer.
        """
        return self.is_fitted

    def __sklearn_tags__(self):
        """Return the scikit-learn tags of the transformer.

        scikit-learn reads these when the transformer is a step of one of its
        meta-estimators. ``requires_fit`` is True for every transformer,
        including those with ``property:fit_is_empty``, since ``transform``
        raises before ``fit``.
        """
        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(preserves_dtype=[]),
            input_tags=InputTags(two_d_array=True, string=True),
            requires_fit=True,
        )

    def _check_X_y(self, X, y):  # noqa: N802
        """Check X and y inputs.

        Coerces X to a pd.DataFrame.
        """
        X = check_X(X)

        if not self.get_tag("capability:multivariate", False) and X.shape[1] > 1:
            raise ValueError(
                f"{type(self).__name__} works on a single column, but X has "
                f"columns {list(X.columns)}. Pass the column name as cols to "
                "pyaptamer.transformations.compose.ApplyToCols(transformer, cols)."
            )

        return X, y

    def _check_X(self, X):  # noqa: N802
        """Check X input.

        Same as _check_X_y but only for X.
        """
        X, _ = self._check_X_y(X, None)
        return X
