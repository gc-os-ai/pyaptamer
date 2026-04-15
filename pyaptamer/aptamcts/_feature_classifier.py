__author__ = ["aditi-dsi"]
__all__ = ["AptaMCTSSequenceEncoder", "AptaMCTSClassifier"]

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class AptaMCTSSequenceEncoder(TransformerMixin, BaseEstimator):
    """
    Converts raw aptamer and protein sequences into numerical feature vectors.

    This transformer applies an encoding strategy to translate string sequences
    into a continuous mathematical space. This step is strictly required before
    passing the data to a standard classifier like Random Forest.

    Parameters
    ----------
    k : int, default=4
        The k-mer window size used for extracting subsequence patterns from
        the RNA and protein strings.

    Attributes
    ----------
    n_features_out_ : int
        The number of numerical features generated after transformation.

    Examples
    --------
    >>> from pyaptamer.aptamcts import AptaMCTSSequenceEncoder
    >>> import numpy as np
    >>> X = np.array([["MKV", "ACGU"], ["MVL", "UGCA"]])
    >>> encoder = AptaMCTSSequenceEncoder(k=4)
    >>> encoder.fit(X)
    >>> encoded_X = encoder.transform(X)
    """

    def __init__(self, k=4):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass


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
    ):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
