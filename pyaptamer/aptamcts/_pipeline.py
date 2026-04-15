__author__ = ["aditi-dsi"]
__all__ = ["AptaMCTSPipeline"]

from skbase.base import BaseObject
from sklearn.base import BaseEstimator


class AptaMCTSPipeline(BaseObject, BaseEstimator):
    """
    AptaMCTS pipeline for aptamer–protein interaction prediction [1]_

    Implements the AptaMCTS pipeline, a machine learning method that takes raw
    RNA and protein sequences, converts them into numerical features using
    sequence encoding, and uses a Random Forest classifier to predict whether
    an aptamer and a protein bind (binary classification).

    This class serves as the main end-to-end pipeline.

    Parameters
    ----------
    k : int, optional, default=4
        The k-mer window size used by the sequence encoder to extract patterns
        from the RNA and protein strings.

    estimator : sklearn-compatible estimator or None, default=None
        Estimator applied after feature extraction. If None, uses `AptaMCTSClassifier`.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The underlying sklearn Pipeline object that handles the sequence encoding
        and classification steps together.

    References
    ----------
    .. [1] Lee, Gwangho, et al. "Predicting aptamer sequences that interact
       with target proteins using an aptamer-protein interaction classifier
       and a Monte Carlo tree search approach." PloS one 16.6 (2021): e0253760.
       https://doi.org/10.1371/journal.pone.0253760.g004
    .. [2] https://github.com/leekh7411/Apta-MCTS

    Examples
    --------
    >>> from pyaptamer.aptamcts import AptaMCTSPipeline
    >>> import numpy as np
    >>> pipe = AptaMCTSPipeline(k=4)
    >>> # X data contains raw biological string pairs (aptamer, protein)
    >>> aptamer_seq = "AGCUUAGCGUAC"
    >>> protein_seq = "ACDEFGHIKLMN"
    >>> X_train_pairs = [(aptamer_seq, protein_seq) for _ in range(4)]
    >>> y_train = np.array([0, 1, 0, 1], dtype=np.float32)
    >>> pipe.fit(X_train_pairs, y_train)  # doctest: +ELLIPSIS
    >>> preds = pipe.predict(X_train_pairs)
    >>> proba = pipe.predict_proba(X_train_pairs)
    """

    def __init__(self, k=4, estimator=None):
        pass

    def _build_pipeline(self):
        pass

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass
