__author__ = ["siddharth7113", "rpgv"]
__all__ = ["AptaComClassifier"]

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from xgboost import XGBClassifier


class AptaComClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost-based binary classifier for aptamer–protein interaction.

    This estimator optionally applies tree-based feature selection
    (``SelectFromModel`` with a ``RandomForestClassifier``) followed by an
    ``XGBClassifier``.  It mirrors the model used in the original AptaCOM
    screening script [1]_.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of boosting rounds for the XGBClassifier.
    max_depth : int, default=9
        Maximum tree depth for XGBClassifier.
    learning_rate : float, default=0.1
        Boosting learning rate (``eta``).
    use_feature_selection : bool, default=True
        If True, a ``SelectFromModel`` step (backed by a RandomForest) is
        prepended to the pipeline for feature selection.
    fs_n_estimators : int, default=300
        Number of trees in the RandomForest used for feature selection.
    fs_max_depth : int, default=9
        Maximum depth for the feature-selection RandomForest.
    threshold : str or float, default="mean"
        Threshold passed to ``SelectFromModel``.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The fitted internal pipeline.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels discovered during ``fit``.

    References
    ----------
    .. [1] AptaCOM screening pipeline — original monolithic script.

    Examples
    --------
    >>> from pyaptamer.aptacom import AptaComClassifier
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((40, 10)).astype(np.float32)
    >>> y = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    >>> clf = AptaComClassifier(random_state=42, verbose=0)
    >>> clf.fit(X, y)  # doctest: +ELLIPSIS
    AptaComClassifier(...)
    >>> preds = clf.predict(X)
    >>> preds.shape
    (40,)
    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=9,
        learning_rate=0.1,
        use_feature_selection=True,
        fs_n_estimators=300,
        fs_max_depth=9,
        threshold="mean",
        random_state=None,
        verbose=0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_feature_selection = use_feature_selection
        self.fs_n_estimators = fs_n_estimators
        self.fs_max_depth = fs_max_depth
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose

    def _build_pipeline(self):
        steps = []

        if self.use_feature_selection:
            selector = SelectFromModel(
                estimator=clone(
                    RandomForestClassifier(
                        n_estimators=self.fs_n_estimators,
                        max_depth=self.fs_max_depth,
                        random_state=self.random_state,
                    )
                ),
                threshold=self.threshold,
            )
            steps.append(("select", selector))

        xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=self.verbose,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        steps.append(("clf", xgb))

        return Pipeline(steps)

    def fit(self, X, y):
        """Fit the classifier on training data.

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

        self.classes_, y = np.unique(y, return_inverse=True)
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(
            X.astype(np.float32, copy=False), y.astype(np.float32, copy=False)
        )
        return self

    def predict(self, X):
        """Predict binary class labels for samples in ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False).astype(np.float32, copy=False)
        y = self.pipeline_.predict(X).astype(int, copy=False)
        return self.classes_[y]

    def predict_proba(self, X):
        """Predict class probabilities for samples in ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Probability estimates for each class.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False).astype(np.float32, copy=False)
        return self.pipeline_.predict_proba(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
