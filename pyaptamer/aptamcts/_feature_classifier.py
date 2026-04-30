__author__ = ["aditi-dsi"]
__all__ = ["AptaMCTSClassifier"]

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class AptaMCTSClassifier(ClassifierMixin, BaseEstimator):
    """
    Random Forest-based binary classifier used as a scoring function for AptaMCTS.

    This model applies a sklearn Random Forest classifier to predict aptamer-protein
    interactions. It only accepts numerical features. RNA and protein sequences
    must already be converted into numbers (e.g., via AptaMCTSSequenceEncoder)
    before passing them in here.

    The estimator is non-deterministic and only supports binary classification.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest. The original implementation used ranges
        between 35 and 200.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split.

    class_weight : {"balanced", "balanced_subsample"}, dict or None, default="balanced"
        Weights associated with classes. "balanced" uses the values of y to
        automatically adjust weights inversely proportional to class frequencies.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the bootstrapping of the samples used
        when building trees.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both `fit` and `predict`.
        `None` means 1 unless in a `joblib.parallel_backend` context.
        `-1` means using all available processors.

    Attributes
    ----------
    estimator_ : sklearn.ensemble.RandomForestClassifier
        The fitted internal random forest model.

    classes_ : ndarray of shape (n_classes,)
        The unique class labels observed in the training data.

    n_features_in_ : int
        Number of features seen during the fit process.

    References
    ----------
    .. [1] Lee, Gwangho, et al. "Predicting aptamer sequences that interact
    with target proteins using an aptamer-protein interaction classifier
    and a Monte Carlo tree search approach." PloS one 16.6 (2021): e0253760.
       https://doi.org/10.1371/journal.pone.0253760.g004
    .. [2] https://github.com/leekh7411/Apta-MCTS

    Examples
    --------
    >>> from pyaptamer.aptamcts import AptaMCTSClassifier
    >>> import numpy as np
    >>> X = np.random.rand(3, 10)
    >>> y = np.array([1, 0, 1])
    >>> clf = AptaMCTSClassifier(n_estimators=50, random_state=42)
    >>> clf.fit(X, y)
    >>> preds = clf.predict(np.random.rand(1, 10))
    >>> proba = clf.predict_proba(np.random.rand(1, 10))
    """

    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        class_weight="balanced",
        random_state=None,
        n_jobs=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary class labels (0/1).

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X, y = validate_data(self, X, y)

        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. Got target type {y_type}."
            )

        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_

        return self

    def predict(self, X):
        """
        Predict binary class labels for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted classes.
        """

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_pred = self.estimator_.predict(X)

        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the `classes_` attribute.
        """

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        proba = self.estimator_.predict_proba(X)

        return proba

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        tags.non_deterministic = True
        return tags
