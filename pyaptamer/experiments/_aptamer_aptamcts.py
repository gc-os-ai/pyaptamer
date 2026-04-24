__all__ = ["AptamerEvalAptaMCTS"]

import numpy as np

from pyaptamer.experiments._aptamer import BaseAptamerEval


class AptamerEvalAptaMCTS(BaseAptamerEval):
    """Candidate aptamer evaluation for a target using AptaMCTS pipeline.

    Parameters
    ----------
    target : str
        Target sequence string.
    pipeline : AptaMCTSPipeline
        Pipeline used to assign interaction scores.
    """

    def __init__(self, target: str, pipeline) -> None:
        super().__init__(target)
        self.pipeline = pipeline

    def evaluate(self, aptamer_candidate: str) -> np.float64:
        """Evaluate aptamer candidate against the target sequence."""
        return np.float64(
            self.pipeline.predict(aptamer=aptamer_candidate, target=self.target)
        )
