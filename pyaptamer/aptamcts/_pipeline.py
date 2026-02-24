"""AptaMCTS pipeline for candidate aptamer recommendation."""

__author__ = ["nennomp", "codex"]
__all__ = ["AptaMCTS"]

import numpy as np
from skbase.base import BaseObject

from pyaptamer.experiments import AptamerEvalAptaNet
from pyaptamer.mcts import MCTS


class AptaMCTS(BaseObject):
    """AptaMCTS pipeline for aptamer recommendation using MCTS and AptaNet scoring.

    The pipeline initializes an `AptamerEvalAptaNet` experiment for a given target
    sequence and uses `MCTS` to recommend high-scoring aptamer candidates.

    Parameters
    ----------
    pipeline : AptaNetPipeline
        Fitted AptaNet pipeline used as scoring function.
    depth : int, optional, default=20
        Target length for generated candidates.
    n_iterations : int, optional, default=1000
        Number of MCTS iterations per round.

    References
    ----------
    .. [1] Lee, Gwangho, et al. "Predicting aptamer sequences that interact with target
       proteins using an aptamer-protein interaction classifier and a Monte Carlo tree
       search approach." PLOS ONE 16.6 (2021): e0253760.
    """

    def __init__(self, pipeline, depth: int = 20, n_iterations: int = 1000) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.depth = depth
        self.n_iterations = n_iterations

    def _init_aptamer_experiment(self, target: str) -> AptamerEvalAptaNet:
        """Initialize aptamer recommendation experiment for `target`."""
        return AptamerEvalAptaNet(target=target, pipeline=self.pipeline)

    def predict(self, candidate: str, target: str) -> np.float64:
        """Score one aptamer candidate against a target protein sequence."""
        experiment = self._init_aptamer_experiment(target=target)
        return experiment.evaluate(candidate)

    def recommend(
        self,
        target: str,
        n_candidates: int = 10,
        verbose: bool = True,
    ) -> set[tuple[str, str, float]]:
        """Recommend aptamer candidates for a target protein sequence."""
        experiment = self._init_aptamer_experiment(target=target)
        mcts = MCTS(
            experiment=experiment,
            depth=self.depth,
            n_iterations=self.n_iterations,
        )

        candidates = set()
        while len(candidates) < n_candidates:
            candidate = mcts.run(verbose=verbose)
            candidates.add(tuple(candidate.values()))

        if verbose:
            for candidate, sequence, score in candidates:
                print(
                    f"Candidate: {candidate}, "
                    f"Sequence: {sequence}, "
                    f"Score: {float(score):.4f}"
                )

        return candidates
