from xgboost import XGBClassifier as xgbc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from pyaptamer.utils._aptacom_utils import pairs_to_features


class AptaComPipeline():
    def __init__(self, features):
        self.features = features #List of feature keys
    
    def _build_pipeline(self):
        transformer = FunctionTransformer(
            func=pairs_to_features,
            kw_args={"features":self.features}
        )