"""Candidate aptamer evaluation with a fitted AptaMCTS pipeline."""

__author__ = ["aditi-dsi"]
__all__ = ["AptamerEvalAptaMCTS"]

import numpy as np

from pyaptamer.data import MoleculeLoader
from pyaptamer.experiments._aptamer import BaseAptamerEval


class AptamerEvalAptaMCTS(BaseAptamerEval):
    """
    Candidate aptamer evaluation for a given target protein using AptaMCTS.

    Evaluates candidate aptamer sequences generated during MCTS against a
    fixed target protein with a fitted ``AptaMCTSPipeline``. The score is the
    predicted probability of the positive class, the second column of
    ``predict_proba``.

    Parameters
    ----------
    target : str
        Target sequence string consisting of amino acid letters. Interpreted
        as the amino-acid sequence of the binding target protein.
    pipeline : AptaMCTSPipeline
        Fitted AptaMCTSPipeline to use for assigning binding probability scores.

    References
    ----------
    .. [1] Lee, Gwangho, et al. "Predicting aptamer sequences that interact
       with target proteins using an aptamer-protein interaction classifier
       and a Monte Carlo tree search approach." PloS one 16.6 (2021): e0253760.
       https://doi.org/10.1371/journal.pone.0253760

    Examples
    --------
    >>> from pyaptamer.aptamcts import AptaMCTSPipeline
    >>> from pyaptamer.datasets import load_li2014
    >>> from pyaptamer.experiments import AptamerEvalAptaMCTS
    >>> X, y = load_li2014(split="test", return_X_y=True)
    >>> pipeline = AptaMCTSPipeline().fit(X, y["label"])
    >>> target = X.to_dataframe()["protein"].iloc[0]
    >>> experiment = AptamerEvalAptaMCTS(target, pipeline)
    >>> score = experiment.evaluate("AUGGC")
    """

    def __init__(self, target: str, pipeline) -> None:
        super().__init__(target)
        self.pipeline = pipeline

    def evaluate(self, aptamer_candidate: str) -> np.float64:
        """
        Evaluate the given aptamer candidate against the target protein.

        The pair is encoded and classified by the pipeline. The score is
        the predicted probability of the positive class.

        Parameters
        ----------
        aptamer_candidate : str
            The aptamer candidate to evaluate. It should be a string consisting of
            letters representing nucleotides.

        Returns
        -------
        np.float64
            The probability score assigned to the aptamer candidate.
        """

        X = MoleculeLoader(
            data={"aptamer": [aptamer_candidate], "protein": [self.target]}
        )
        score = self.pipeline.predict_proba(X=X)

        return np.float64(score[:, 1].item())
