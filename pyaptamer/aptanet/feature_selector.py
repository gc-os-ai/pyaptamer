__author__ = "satvshr"
__all__ = ["FeatureSelector"]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import check_is_fitted, validate_data


class FeatureSelector(TransformerMixin, BaseEstimator):
    """
    Feature selector using a Random Forest classifier to identify and retain
    important features based on feature importances.

    This transformer can be used in scikit-learn pipelines to perform automatic
    feature selection as a preprocessing step before model training.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees in the random forest. Default is 300.

    max_depth : int, optional
        Maximum depth of the trees. Default is 9.

    random_state : int, optional
        Seed used by the random number generator. Default is 0.

    Attributes
    ----------
    clf_ : RandomForestClassifier
        Fitted random forest classifier used to compute feature importances.

    clf_model_ : SelectFromModel
        Model used to select features based on importances from the fitted classifier.
    """

    def __init__(self, n_estimators=300, max_depth=9, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.clf_.fit(X, y)
        self.clf_model_ = SelectFromModel(self.clf_, prefit=True)
        return self

    def transform(self, X):
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)

        if not hasattr(self, "clf_model_"):
            raise ValueError("Feature selector has not been fitted yet.")

        return self.clf_model_.transform(X)
