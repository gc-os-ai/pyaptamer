__author__ = ["siddharth7113", "rpgv"]
__all__ = ["AptaComPipeline"]

from skbase.base import BaseObject
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from pyaptamer.aptacom._aptacom import AptaComClassifier
from pyaptamer.utils._aptacom_utils import pairs_to_features


class AptaComPipeline(BaseObject, BaseEstimator):
    """AptaCOM algorithm for aptamer–protein interaction prediction.

    Implements the AptaCOM pipeline, an XGBoost-based method that combines
    PyBioMed physicochemical features (DNA and protein descriptors),
    optional secondary-structure dot-bracket analysis, and optional
    protein SASA features to predict whether an aptamer and a protein
    interact (binary classification).

    The pipeline starts from tuples of strings, converts them into numeric
    features via ``pairs_to_features``, and feeds the result into the
    estimator.

    Parameters
    ----------
    features : list of str, optional, default=None
        Ordered list of feature keys describing the contents of each input
        tuple.  Valid keys are ``"aptamer"``, ``"target"``, ``"ss"``,
        ``"pdb_id"``.  If ``None``, defaults to ``["aptamer", "target"]``.

    estimator : sklearn-compatible estimator or None, default=None
        Estimator applied after feature extraction.  If ``None``, uses
        ``AptaComClassifier`` with default hyper-parameters.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The underlying sklearn Pipeline that handles feature extraction
        and classification.

    Examples
    --------
    >>> from pyaptamer.aptacom import AptaComPipeline
    >>> import numpy as np
    >>> apt = "AGTCGATGGCTGAGGGATCGATG"
    >>> trg = "MWLGRRALCALVLLLACASLGLLYASTRDAPGLRL"
    >>> X = [(apt, trg) for _ in range(40)]
    >>> y = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    >>> pipe = AptaComPipeline()
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    >>> preds = pipe.predict(X)
    """

    def __init__(self, features=None, estimator=None):
        self.features = features
        self.estimator = estimator

    def _build_pipeline(self):
        features = self.features or ["aptamer", "target"]

        transformer = FunctionTransformer(
            func=pairs_to_features,
            kw_args={"features": features},
            validate=False,
        )
        self._estimator = self.estimator or AptaComClassifier()
        return Pipeline([("features", transformer), ("clf", clone(self._estimator))])

    def fit(self, X, y):
        """Fit the pipeline on training data.

        Parameters
        ----------
        X : list of tuple of str
            Each tuple contains string-valued inputs in the order described
            by ``features``.  For example, with the default feature set each
            element should be ``(aptamer_seq, target_seq)``.
        y : array-like of shape (n_samples,)
            Binary class labels (0/1).

        Returns
        -------
        self : object
            Fitted pipeline.
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        """Predict class labels for samples in ``X``.

        Parameters
        ----------
        X : list of tuple of str
            Input data in the same format as ``fit``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for samples in ``X``.

        Parameters
        ----------
        X : list of tuple of str
            Input data in the same format as ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Probability estimates for each class.
        """
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)
