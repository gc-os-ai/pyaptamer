__author__ = "satvshr"
__all__ = ["AptaNetPipeline"]
__required__ = ["python>=3.9,<3.12"]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from pyaptamer.aptanet._feature_classifier import AptaNetFeaturesClassifier
from pyaptamer.utils._aptanet_utils import pairs_to_features


class AptaNetPipeline:
    """
    Pipeline for aptamer-pair classification that starts from **string pairs**,
    converts them to numeric features with `pairs_to_features`, then applies
    tree-based feature selection and a skorch-wrapped neural network
    (`AptaNetFeaturesClassifier`).

    This class wraps an internal scikit-learn `Pipeline` and delegates `fit`,
    `predict`, and related methods to it.

    References
    ----------


    - Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
    aptamer–protein interaction prediction. Sci Rep 11, 6074 (2021).
    https://doi.org/10.1038/s41598-021-85629-0
    - https://github.com/nedaemami/AptaNet
    - https://www.nature.com/articles/s41598-021-85629-0.pdf


    Parameters
    ----------
    pairs_to_features_kwargs : dict, default=None
        Extra keyword arguments passed directly to `pairs_to_features`.


        - k : int, optional, default=4
        The k-mer size used to generate aptamer k-mer vectors.
        - pseaac_kwargs : dict, optional, default=None
        Keyword arguments forwarded to PseAAC feature extraction, e.g.
        {"lambda_value": 30, "w": 0.05}.


    **features_classifier_kwargs : dict
        Keyword arguments forwarded to `AptaNetFeaturesClassifier` to configure
        the feature selector and neural network. See that class’ docstring for
        the full list and defaults.

    Examples
    --------
    >>> from pyaptamer.aptanet.pipeline import AptaNetPipeline
    >>> import numpy as np
    >>> pipe = AptaNetPipeline()
    >>> aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> X_train_pairs = [(aptamer_seq, protein_seq) for _ in range(40)]
    >>> y_train = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    >>> X_test_pairs = [(aptamer_seq, protein_seq) for _ in range(10)]
    >>> pipe.fit(X_train_pairs, y_train)
    >>> preds = pipe.predict(X_test_pairs)
    """

    def __init__(self, pairs_to_features_kwargs=None, **features_classifier_kwargs):
        self.pairs_to_features_kwargs = pairs_to_features_kwargs
        self.features_classifier_kwargs = features_classifier_kwargs

    def _build_pipeline(self):
        transformer = FunctionTransformer(
            func=pairs_to_features,
            kw_args=self.pairs_to_features_kwargs,
            validate=False,
        )
        clf = AptaNetFeaturesClassifier(**self.features_classifier_kwargs)
        return Pipeline([("features", transformer), ("clf", clf)])

    def fit(self, X, y):
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)

    def predict(self, X):
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("Pipeline not fitted. Call fit() before predict().")
        return self.pipeline_.predict(X)
