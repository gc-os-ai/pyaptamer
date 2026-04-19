__author__ = "Jayant-kernel"
__all__ = ["AptaComPipeline"]

from skbase.base import BaseObject
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from pyaptamer.aptacom._aptacom_utils import aptacom_pairs_to_features


class AptaComPipeline(BaseObject, BaseEstimator):
    """
    AptaCom DNA feature extractor with XGBoost classifier [1]_.

    Implements the AptaCom DNA feature extraction pipeline, which combines
    comprehensive DNA physicochemical descriptors (DAC, DCC, DACC, TAC, TCC,
    TACC, Kmer, PseDNC, PseKNC, SCPseDNC, SCPseTNC) with an XGBoost
    classifier to predict aptamer properties or interactions.

    The pipeline accepts raw DNA sequences, extracts all AptaCom DNA features,
    and feeds the result into the estimator.

    Parameters
    ----------
    k : int, optional, default=4
        Maximum k-mer size for Kmer and PseKNC feature extraction.
    lag : int, optional, default=1
        Lag parameter for auto/cross-covariance and pseudo-composition features.
    estimator : sklearn-compatible estimator or None, default=None
        Classifier used after feature extraction. If None, defaults to
        ``XGBClassifier`` with the hyperparameters from the original AptaCom
        paper (n_estimators=100, max_depth=6, learning_rate=0.1).
    random_state : int or None, default=None
        Random seed passed to the default ``XGBClassifier`` when no custom
        estimator is provided.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The fitted sklearn Pipeline (set after calling ``fit``).

    References
    ----------
    .. [1] Emami, N., et al. "AptaCom: Prediction of Aptamer-Protein Interaction
       using Complementary Features and XGBoost Algorithm." *Briefings in
       Bioinformatics*, 2022. https://doi.org/10.1093/bib/bbac415
    .. [2] GitHub repository: https://github.com/rpgv/AptaCom

    Examples
    --------
    >>> from pyaptamer.aptacom import AptaComPipeline
    >>> import numpy as np
    >>> pipe = AptaComPipeline()
    >>> seqs = ["AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"] * 40
    >>> y = np.array([0] * 20 + [1] * 20)
    >>> pipe.fit(seqs, y)  # doctest: +ELLIPSIS
    AptaComPipeline(...)
    >>> preds = pipe.predict(seqs)
    >>> proba = pipe.predict_proba(seqs)
    """

    def __init__(self, k=4, lag=1, estimator=None, random_state=None):
        self.k = k
        self.lag = lag
        self.estimator = estimator
        self.random_state = random_state

    def _build_pipeline(self):
        from xgboost import XGBClassifier

        transformer = FunctionTransformer(
            func=aptacom_pairs_to_features,
            kw_args={"k": self.k, "lag": self.lag},
            validate=False,
        )
        estimator = self.estimator or XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=self.random_state,
        )
        return Pipeline([("features", transformer), ("clf", clone(estimator))])

    def fit(self, X, y):
        """
        Fit the AptaCom pipeline on training data.

        Parameters
        ----------
        X : list of str or array-like
            A list of DNA sequences (strings).
        y : array-like of shape (n_samples,)
            Binary class labels (0/1).

        Returns
        -------
        self : AptaComPipeline
            Fitted estimator.
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict binary class labels for DNA sequences in X.

        Parameters
        ----------
        X : list of str or array-like
            A list of DNA sequences (strings).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for DNA sequences in X.

        Parameters
        ----------
        X : list of str or array-like
            A list of DNA sequences (strings).

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probability estimates. Column 0 is P(non-binding),
            column 1 is P(binding).
        """
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)
