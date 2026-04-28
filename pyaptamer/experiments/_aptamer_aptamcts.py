__all__ = ["AptamerEvalAptaMCTS"]

import numpy as np

from pyaptamer.experiments._aptamer import BaseAptamerEval


class AptamerEvalAptaMCTS(BaseAptamerEval):
    """Candidate aptamer evaluation for a given target protein using AptaMCTS.

    Parameters
    ----------
    target : str
        Target sequence string consisting of amino acid letters and (possibly) unknown
        characters. Interpreted as the amino-acid sequence of the binding target
        protein.
    pipeline : AptaMCTSPipeline
        Fitted AptaMCTSPipeline to use for assigning scores.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptamcts import AptaMCTSPipeline
    >>> from pyaptamer.experiments import AptamerEvalAptaMCTS
    >>> aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> pairs = [(aptamer_seq, protein_seq) for _ in range(40)]
    >>> labels = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    >>> pipeline = AptaMCTSPipeline()
    >>> pipeline.fit(pairs, labels)  # doctest: +ELLIPSIS
    AptaMCTSPipeline(...)
    >>> target = "ACDEFACDEFACDEFACDEFACDEFACDEFACDEFACDEF"
    >>> experiment = AptamerEvalAptaMCTS(target, pipeline)
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
        score = self.pipeline.predict_proba(X=[(aptamer_candidate, self.target)])

        return np.float64(score[:, 1].item())  # return the positive class probability
