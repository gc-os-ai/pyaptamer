__author__ = "satvshr"
__all__ = ["AptaNetPipeline"]
__required__ = ["python>=3.9,<3.12"]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from pyaptamer.aptanet._feature_classifier import AptaNetFeaturesClassifier
from pyaptamer.utils._aptanet_utils import pairs_to_features


class AptaNetPipeline:
    """
    Convenience wrapper that accepts **string pairs** (raw aptamer pairs),
    converts them to numeric features with `pairs_to_features`, and then
    delegates to `AptaNetFeaturesClassifier`.

    This is *not* intended to be run through sklearn's estimator checks.
    """

    def __init__(self, pairs_to_features_kwargs=None, **features_classifier_kwargs):
        self.pairs_to_features_kwargs = pairs_to_features_kwargs
        self.features_classifier_kwargs = features_classifier_kwargs

    def _build_pipeline(self):
        transformer = FunctionTransformer(
            func=pairs_to_features,
            kw_args=self.pairs_to_features_kwargs,
            validate=False,  # allow raw strings through
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
