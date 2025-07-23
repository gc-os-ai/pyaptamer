from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import check_is_fitted, validate_data


class FeatureSelector(TransformerMixin, BaseEstimator):
    """
    Feature selector using Random Forest to select important features from the input
    data.

    This class fits a Random Forest model to the training data and selects features
    based on their importance scores. It is used to reduce the dimensionality of the
    input data before training AptaNetMLP.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees in the forest (default is 300).
    max_depth : int, optional
        Maximum depth of the tree (default is 9).
    random_state : int, optional
        Random seed for reproducibility (default is 0).
    Methods
    -------
    fit(X, y)
        Fit the feature selector to the training data.
    transform(X)
        Transform the input data by selecting important features.
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
