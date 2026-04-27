__author__ = ["nennomp", "satvshr", "siddharth7113"]
__all__ = ["AptaNetPipeline"]
__required__ = ["python>=3.10"]

from skbase.base import BaseObject
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from pyaptamer.aptanet import AptaNetClassifier
from pyaptamer.utils._aptanet_utils import pairs_to_features


class AptaNetPipeline(BaseObject, BaseEstimator):
    """AptaNet algorithm for aptamer-protein interaction prediction [1]_.

    Implements the AptaNet algorithm, a deep learning method that combines
    sequence-derived features with RandomForest-based feature selection and a
    multi-layer perceptron to predict whether an aptamer and a protein interact
    (binary classification).

    The pipeline starts from string pairs, converts them into numeric features
    (aptamer k-mer frequencies + protein PSeAAC), applies tree-based feature
    selection, and feeds the result into the estimator.

    ``fit``, ``predict``, and ``predict_proba`` accept any input shape
    supported by ``APIDataset.from_any``:

    - ``list[tuple[str, str]]`` of (aptamer, protein) pairs
    - ``pd.DataFrame`` with columns ``["aptamer", "protein"]``
    - ``tuple[np.ndarray, np.ndarray]`` of (aptamer_array, protein_array)
    - ``tuple[MoleculeLoader, MoleculeLoader]``
    - ``APIDataset`` instance

    Parameters
    ----------
    k : int, optional, default=4
        The k-mer size used to generate aptamer k-mer vectors.

    estimator : sklearn-compatible estimator or None, default=None
        Estimator applied after feature selection. If None, uses
        ``AptaNetClassifier``.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The underlying sklearn Pipeline object that handles feature extraction,
        feature selection, and classification.

    References
    ----------
    .. [1] Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
        aptamer-protein interaction prediction. *Scientific Reports*, 11,
        6074 (2021). https://doi.org/10.1038/s41598-021-85629-0
    .. [2] GitHub repository: https://github.com/nedaemami/AptaNet

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
    >>> _ = pipe.fit(X_train_pairs, y_train)
    >>> preds = pipe.predict(X_test_pairs)
    >>> proba = pipe.predict_proba(X_test_pairs)
    """

    def __init__(self, k=4, estimator=None):
        self.k = k
        self.estimator = estimator

    def _build_pipeline(self):
        transformer = FunctionTransformer(
            func=pairs_to_features,
            kw_args={"k": self.k},
            validate=False,
        )
        self._estimator = self.estimator or AptaNetClassifier()
        return Pipeline([("features", transformer), ("clf", clone(self._estimator))])

    @staticmethod
    def _to_pairs(X, y=None):
        """Coerce X to list-of-tuples for the internal sklearn pipeline."""
        from pyaptamer.datasets.dataclasses import APIDataset

        ds = APIDataset.from_any(X, y)
        df = ds.load()
        pairs = list(zip(df["aptamer"].tolist(), df["protein"].tolist(), strict=False))
        return pairs, ds.y

    def fit(self, X, y=None):
        pairs, y_out = self._to_pairs(X, y)
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(pairs, y_out)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        pairs, _ = self._to_pairs(X)
        return self.pipeline_.predict_proba(pairs)

    def predict(self, X):
        check_is_fitted(self)
        pairs, _ = self._to_pairs(X)
        return self.pipeline_.predict(pairs)
