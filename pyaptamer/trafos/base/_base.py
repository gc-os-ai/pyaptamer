"""Base transformation class."""

__author__ = ["fkiraly", "siddharth7113"]

import pandas as pd
from skbase.base import BaseEstimator
from sklearn.utils import InputTags, Tags, TargetTags, TransformerTags

from pyaptamer.data import MoleculeLoader


class BaseTransform(BaseEstimator):
    """Base class for all transformations."""

    _tags = {
        "object_type": "transformer",
        # "input_type":
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
        X : array-like, shape (n_samples, n_features)
            Input data to fit the transformer.
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
        X : array-like, shape (n_samples, n_features)
            Input data to fit the transformer.
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
        X : array-like, shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X : array-like, shape (n_samples, n_features_transformed)
            Transformed data.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called before.
        """
        self.check_is_fitted(method_name="transform")
        X_inner = self._check_X(X)
        Xt = self._transform(X=X_inner)
        return Xt

    def _transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to transform.

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

    def __sklearn_tags__(self):
        """Return the sklearn tags of this transformer.

        ``BaseTransform`` extends skbase's ``BaseEstimator``,so
        sklearn has no ``__sklearn_tags__`` to inherit. This adds the tags
        directly, so meta-estimators such as ``sklearn.pipeline.Pipeline`` can
        introspect transformers.

        Returns
        -------
        tags : sklearn.utils.Tags
            Tags describing this transformer to sklearn.
        """
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(),
            classifier_tags=None,
            regressor_tags=None,
            # X is a MoleculeLoader or a pd.DataFrame of sequence strings,
            # never a numeric 2D array
            input_tags=InputTags(two_d_array=False, string=True),
        )

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
                f"Transformer {type(self).__name__} only supports univariate data, "
                f"but X has {X.shape[1]} features."
            )

        return X, y

    def _check_X(self, X):  # noqa: N802
        """Check X input.

        Same as _check_X_y but only for X.
        """
        X, _ = self._check_X_y(X, None)
        return X
