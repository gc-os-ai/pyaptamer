__all__ = ["AptaMCTSPipeline"]

import numpy as np

from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.mcts import MCTS
from pyaptamer.utils._aptamcts_utils import pairs_to_features


class AptaMCTSPipeline:
    """AptaMCTS pipeline for aptamer recommendation.

    This pipeline wraps a pre-trained model exposing ``predict_proba`` and combines
    it with Monte Carlo Tree Search (MCTS) for candidate recommendation.

    Parameters
    ----------
    model : object
        Pre-trained model exposing a ``predict_proba`` method.
    depth : int, optional, default=20
        Search depth passed to MCTS.
    n_iterations : int, optional, default=1000
        Number of iterations passed to MCTS.
    """

    def __init__(self, model, depth=20, n_iterations=1000):
        self.model = model
        self.depth = depth
        self.n_iterations = n_iterations

    def _init_aptamer_experiment(self, target: str) -> AptamerEvalAptaMCTS:
        """Initialize the aptamer recommendation experiment."""
        return AptamerEvalAptaMCTS(target=target, pipeline=self)

    def predict(self, aptamer: str, target: str) -> np.float64:
        """Predict interaction score for an aptamer-target pair.

        Parameters
        ----------
        aptamer : str
            Aptamer candidate sequence.
        target : str
            Target sequence.

        Returns
        -------
        np.float64
            Positive-class interaction score.
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("`model` must implement `predict_proba`.")

        features = pairs_to_features([(aptamer, target)])
        score = self.model.predict_proba(features)

        if score.ndim != 2 or score.shape[1] < 2:
            raise ValueError(
                "`predict_proba` must return an array with shape "
                "(n_samples, n_classes>=2)."
            )

        return np.float64(score[:, 1].item())

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
        experiment = self._init_aptamer_experiment(target=target)
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
