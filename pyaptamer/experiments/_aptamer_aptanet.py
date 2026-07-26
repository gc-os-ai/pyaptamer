__all__ = ["AptamerEvalAptaNet"]

import numpy as np

from pyaptamer.data import MoleculeLoader
from pyaptamer.experiments._aptamer import BaseAptamerEval


class AptamerEvalAptaNet(BaseAptamerEval):
    """Candidate aptamer evaluation for a given target protein using AptaNet.

    Parameters
    ----------
    target : str
        Target sequence string consisting of amino acid letters and (possibly) unknown
        characters. Interpreted as the amino-acid sequence of the binding target
        protein.
    pipeline : AptaNetPipeline
        Fitted AptaNetPipeline to use for assigning scores.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptanet import AptaNetPipeline
    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.experiments import AptamerEvalAptaNet
    >>> aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> X = MoleculeLoader(
    ...     data={"aptamer": [aptamer_seq] * 5, "protein": [protein_seq] * 5}
    ... )
    >>> labels = np.array([0] * 5, dtype=np.float32)
    >>> pipeline = AptaNetPipeline()
    >>> pipeline.fit(X, labels)  # doctest: +ELLIPSIS
    AptaNetPipeline(...)
    >>> target = "ACDEFACDEFACDEFACDEFACDEFACDEFACDEFACDEF"
    >>> experiment = AptamerEvalAptaNet(target, pipeline)
    >>> aptamer_candidate = "AUGGC"
    >>> score = experiment.evaluate(aptamer_candidate)
    """

    def __init__(self, target: str, pipeline) -> None:
        super().__init__(target)
        self.pipeline = pipeline

    def evaluate(self, aptamer_candidate: str) -> np.float64:
        """Evaluate the given aptamer candidate against the target protein.

        Parameters
        ----------
        aptamer_candidate : str
            The aptamer candidate to evaluate. It should be a string consisting of
            letters representing nucleotides: 'A', 'C', 'G', and 'U' (for RNA) or 'T'
            (for DNA).

        Returns
        -------
        np.float64
            The probability score assigned to the aptamer candidate.
        """
        X = MoleculeLoader(
            data={"aptamer": [aptamer_candidate], "protein": [self.target]}
        )
        score = self.pipeline.predict_proba(X=X)

        return np.float64(score[:, 1].item())  # return the positive class probability
