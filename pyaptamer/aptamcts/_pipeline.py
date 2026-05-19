__author__ = ["agastya"]
__all__ = ["AptaMCTSPipeline"]

from skbase.base import BaseObject
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.mcts import MCTS
from pyaptamer.utils._aptamcts_utils import pairs_to_features


class AptaMCTSPipeline(BaseObject, BaseEstimator):
    """AptaMCTS pipeline for aptamer–target interaction prediction.

    Implements a sklearn-style pipeline that converts aptamer–target sequence pairs
    into numerical features using a placeholder iCTF-style encoder, applies feature
    selection, and feeds the result into a classifier for binding prediction.

    The pipeline starts from string pairs, converts them into numeric features
    (aptamer nucleotide frequencies + protein PSeAAC), and passes them to the
    estimator.

    Parameters
    ----------
    estimator : sklearn-compatible classifier or None, default=None
        Estimator applied after feature extraction. Must implement ``predict_proba``
        for binary classification. If None, a default classifier is used.
    depth : int, optional, default=20
        Search depth passed to MCTS during recommendation.
    n_iterations : int, optional, default=1000
        Number of iterations passed to MCTS during recommendation.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The underlying sklearn Pipeline object that handles feature extraction
        and classification.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptamcts import AptaMCTSPipeline
    >>> pipe = AptaMCTSPipeline()
    >>> aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> X_train = [(aptamer_seq, protein_seq) for _ in range(40)]
    >>> y_train = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    >>> pipe.fit(X_train, y_train)  # doctest: +ELLIPSIS
    AptaMCTSPipeline(...)
    >>> X_test = [(aptamer_seq, protein_seq) for _ in range(10)]
    >>> preds = pipe.predict(X_test)
    >>> proba = pipe.predict_proba(X_test)
    """

    def __init__(self, estimator=None, depth=20, n_iterations=1000):
        self.estimator = estimator
        self.depth = depth
        self.n_iterations = n_iterations

    def _build_pipeline(self):
        transformer = FunctionTransformer(
            func=pairs_to_features,
            validate=False,
        )
        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            self._estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self._estimator = clone(self.estimator)
        return Pipeline([("features", transformer), ("clf", self._estimator)])

    def fit(self, X, y):
        """Fit the pipeline on aptamer–target pairs and labels.

        Parameters
        ----------
        X : list[tuple[str, str]]
            Sequence pairs ``(aptamer, target)``.
        y : array-like
            Binary labels (0 = no binding, 1 = binding).

        Returns
        -------
        self
            Fitted pipeline.
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        return self

    def predict_proba(self, X):
        """Predict class probabilities for aptamer–target pairs.

        Parameters
        ----------
        X : list[tuple[str, str]]
            Sequence pairs ``(aptamer, target)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, n_classes)`` with class probabilities.
        """
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)

    def predict(self, X):
        """Predict binding class labels for aptamer–target pairs.

        Parameters
        ----------
        X : list[tuple[str, str]]
            Sequence pairs ``(aptamer, target)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples,)`` with predicted class labels.
        """
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    def recommend(self, target: str, n_candidates=10):
        """Recommend aptamer candidates for a target using MCTS.

        Parameters
        ----------
        target : str
            Target sequence.
        n_candidates : int, optional, default=10
            Number of unique candidates to return.

        Returns
        -------
        set[tuple[str, str, float]]
            Set of ``(candidate, sequence, score)`` tuples.
        """
        check_is_fitted(self)
        experiment = AptamerEvalAptaMCTS(target=target, pipeline=self)
        mcts = MCTS(
            experiment=experiment,
            depth=self.depth,
            n_iterations=self.n_iterations,
        )

        candidates = {}
        while len(candidates) < n_candidates:
            result = mcts.run(verbose=False)
            candidate = result["candidate"]
            sequence = result["sequence"]
            score = result["score"]

            if candidate not in candidates:
                if hasattr(score, "item"):
                    score = score.item()
                candidates[candidate] = (candidate, sequence, float(score))

        return set(candidates.values())
