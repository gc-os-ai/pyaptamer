"""Base class for end-to-end estimators that wrap a scikit-learn pipeline."""

__author__ = ["siddharth7113"]
__all__ = ["BasePipelineDelegator"]

from skbase.base import BaseEstimator
from sklearn.utils import InputTags, Tags, get_tags

from pyaptamer.utils._check_x import check_X


class BasePipelineDelegator(BaseEstimator):
    """Base class for estimators that delegate to a scikit-learn pipeline.

    A subclass holds the parameters of one algorithm and builds the inner
    pipeline in ``_build_pipeline``. This class turns the input into a
    DataFrame of the configured columns and passes ``fit``, ``predict``,
    ``predict_proba`` and ``score`` on to that pipeline. The estimator type
    seen by scikit-learn, classifier or regressor, is that of the last step.

    A subclass sets the ``col_params`` tag to the names of its column
    parameters, for example ``["aptamer_col", "protein_col"]``, and
    implements ``_build_pipeline``. Only the columns named by these
    parameters reach the inner pipeline.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The fitted inner pipeline.
    """

    _tags = {
        "object_type": "estimator",
        "col_params": [],
    }

    def _build_pipeline(self):
        """Return the unfitted inner pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline whose last step is the estimator.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_pipeline."
        )

    def _check_X(self, X):  # noqa: N802
        cols = {p: getattr(self, p) for p in self.get_tag("col_params")}
        return check_X(X, cols=cols, owner=type(self).__name__)

    def fit(self, X, y):
        """Fit the inner pipeline.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Training data with the configured columns.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(self._check_X(X), y)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict with the inner pipeline.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Data with the configured columns.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels or values.
        """
        self.check_is_fitted(method_name="predict")
        return self.pipeline_.predict(self._check_X(X))

    def predict_proba(self, X):
        """Predict class probabilities with the inner pipeline.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Data with the configured columns.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Probability estimates, in the order of ``classes_``.
        """
        self.check_is_fitted(method_name="predict_proba")
        return self.pipeline_.predict_proba(self._check_X(X))

    def score(self, X, y):
        """Return the score of the inner pipeline, accuracy for a classifier.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Data with the configured columns.
        y : array-like of shape (n_samples,)
            True labels or values.

        Returns
        -------
        float
            The estimator's score.
        """
        self.check_is_fitted(method_name="score")
        return self.pipeline_.score(self._check_X(X), y)

    @property
    def classes_(self):
        """Class labels seen in ``fit``, for a classifier estimator."""
        return self.pipeline_.classes_

    def __sklearn_tags__(self):
        """Return the scikit-learn tags, with the type of the last step."""
        estimator = get_tags(self._build_pipeline()[-1])
        return Tags(
            estimator_type=estimator.estimator_type,
            target_tags=estimator.target_tags,
            classifier_tags=estimator.classifier_tags,
            regressor_tags=estimator.regressor_tags,
            input_tags=InputTags(two_d_array=True, string=True),
        )
