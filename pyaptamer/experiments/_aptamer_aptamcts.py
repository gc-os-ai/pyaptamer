__all__ = ["AptamerEvalAptaMCTS"]

import numpy as np

from pyaptamer.data import MoleculeLoader
from pyaptamer.experiments._aptamer import BaseAptamerEval


class AptamerEvalAptaMCTS(BaseAptamerEval):
    """
    Candidate aptamer evaluation for a given target protein using AptaMCTS.

    This class serves as a wrapper to evaluate candidate aptamer sequences generated
    during MCTS against a fixed target protein, using the Random Forest-based
    AptaMCTS scoring pipeline.

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
       https://doi.org/10.1371/journal.pone.0253760.g004

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptamcts import AptaMCTSPipeline
    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.experiments import AptamerEvalAptaMCTS
    >>> aptamer_seq = "AGCUUAGCGUAC"
    >>> protein_seq = "ACDEFGHIKLMN"
    >>> X = MoleculeLoader(
    ...     data={"aptamer": [aptamer_seq] * 5, "protein": [protein_seq] * 5}
    ... )
    >>> labels = np.array([0, 1, 0, 1, 0])
    >>> pipeline = AptaMCTSPipeline()
    >>> pipeline.fit(X, labels)  # doctest: +ELLIPSIS
    AptaMCTSPipeline(...)
    >>> target_protein = "ACDEFGHIKLMN"
    >>> experiment = AptamerEvalAptaMCTS(target_protein, pipeline)
    >>> aptamer_candidate = "AUGGC"
    >>> score = experiment.evaluate(aptamer_candidate)
    """

    def __init__(self, target: str, pipeline) -> None:
        super().__init__(target)
        self.pipeline = pipeline

    def evaluate(self, aptamer_candidate: str) -> np.float64:
        """
        Evaluate the given aptamer candidate against the target protein.

        The evaluation involves converting the aptamer-protein pair into
        iCTF features and passing them through the Random Forest classifier
        to obtain the binding probability.

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
