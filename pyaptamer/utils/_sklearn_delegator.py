"""Utility for creating sklearn estimators that are thin wrappers around
an internal ``sklearn.pipeline.Pipeline``.

This is useful when you want to expose a customised API but rely on
existing sklearn components internally. The delegator takes care of building
and fitting the pipeline and forwards attribute access to it once fitted.

Classes defined here should be imported from their public modules (e.g.,
``from pyaptamer.utils import SklearnPipelineDelegator``) rather than
accessing the private module directly.
"""
from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class SklearnPipelineDelegator(BaseEstimator):
    """Base class for estimators delegating to an internal ``Pipeline``.

    Subclasses must implement ``_build_pipeline(self)`` which returns an
    unfitted :class:`sklearn.pipeline.Pipeline` object. ``fit`` will
    construct and fit that pipeline. After fitting, any attribute or method
    lookup not found on the wrapper class itself is delegated to the underlying
    pipeline instance (``self.pipeline_``) via ``__getattr__``.

    The delegator does **not** automatically expose all pipeline methods in its
    signature – it simply forwards calls. This keeps subclasses free to
    override or wrap particular methods (e.g. to add input validation).
    """

    def fit(self, X, y=None, **fit_params: Any) -> "SklearnPipelineDelegator":
        """Build and fit the underlying pipeline.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target values (default: None).
        **fit_params
            Additional parameters forwarded to the pipeline's ``fit`` method.

        Returns
        -------
        self
        """
        # build fresh pipeline on each fit call
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y, **fit_params)
        return self

    # We intentionally avoid implementing ``predict``/``predict_proba`` etc. here
    # so that subclasses may implement custom wrappers if necessary.  Any such
    # method call will be forwarded by __getattr__ when the subclass does not
    # define it.

    def __getattr__(self, name: str) -> Any:
        # this is invoked only if the attribute is not found the normal way
        pipeline = self.__dict__.get("pipeline_")
        if pipeline is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(pipeline, name)

    # convenience helper that subclasses can call to validate fitting
    def _check_fitted(self):
        """Raise a ``NotFittedError`` if pipeline isn't available yet."""
        check_is_fitted(self, "pipeline_")
