__author__ = "satvshr"
__all__ = ["AptaNetPipeline"]
__required__ = ["python>=3.9,<3.13"]

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from pyaptamer.aptanet import AptaNetClassifier
from pyaptamer.utils._aptanet_utils import pairs_to_features


class AptaNetPipeline:
    """
    AptaNetPipeline: AptaNet algorithm for aptamer–protein interaction
    prediction (Emami et al., 2021)

    Implements the AptaNet algorithm, a deep learning method that combines
    sequence-derived features with RandomForest-based feature selection and a
    multi-layer perceptron to predict whether an aptamer and a protein interact
    (binary classification).

    The pipeline starts from string pairs, converts them into numeric features
    (aptamer k-mer frequencies + protein PSeAAC), applies tree-based feature
    selection, and feeds the result into the classifier.

    References
    ----------


    - Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
    aptamer–protein interaction prediction. Sci Rep 11, 6074 (2021).
    https://doi.org/10.1038/s41598-021-85629-0
    - https://github.com/nedaemami/AptaNet
    - https://www.nature.com/articles/s41598-021-85629-0.pdf


    Parameters
    ----------
    k : int, optional, default=4
        The k-mer size used to generate aptamer k-mer vectors.

    classifier : sklearn-compatible estimator or None, default=None
        Estimator applied after feature selection. If None, uses `AptaNetClassifier`.

    Examples
    --------
    >>> from pyaptamer.aptanet import AptaNetPipeline
    >>> import numpy as np
    >>> pipe = AptaNetPipeline()
    >>> aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> X_train_pairs = [(aptamer_seq, protein_seq) for _ in range(40)]
    >>> y_train = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    >>> X_test_pairs = [(aptamer_seq, protein_seq) for _ in range(10)]
    >>> pipe.fit(X_train_pairs, y_train)  # doctest: +ELLIPSIS
    >>> preds = pipe.predict(X_test_pairs)
    """

    def __init__(self, k=None, classifier=None):
        self.k = k
        self.classifier = classifier or AptaNetClassifier()

    def _build_pipeline(self):
        transformer = FunctionTransformer(
            func=pairs_to_features,
            kw_args=self.k,
            validate=False,
        )
        return Pipeline([("features", transformer), ("clf", clone(self.classifier))])

    def fit(self, X, y):
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        return self.pipeline_.predict(X)
