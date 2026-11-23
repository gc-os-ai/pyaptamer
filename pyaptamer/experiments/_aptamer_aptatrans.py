__all__ = ["AptamerEvalAptaTrans"]

import numpy as np

from pyaptamer.data import MoleculeLoader
from pyaptamer.experiments._aptamer import BaseAptamerEval


class AptamerEvalAptaTrans(BaseAptamerEval):
    """Candidate aptamer evaluation for a given target protein using AptaTrans.

    Parameters
    ----------
    target : str
        Target sequence string consisting of amino acid letters and (possibly) unknown
        characters. Interpreted as the amino-acid sequence of the binding target
        protein.
    pipeline : AptaTransPipeline
        Fitted AptaTransPipeline to use for assigning scores. The pair is built
        with its ``aptamer_col`` and ``protein_col``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptatrans import AptaTransClassifier, AptaTransPipeline
    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.experiments import AptamerEvalAptaTrans
    >>> aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> X = MoleculeLoader(data={"aptamer": [aptamer] * 8, "protein": [protein] * 8})
    >>> y = np.array([0, 1] * 4)
    >>> clf = AptaTransClassifier(
    ...     apta_max_len=16,
    ...     in_dim=8,
    ...     n_heads=2,
    ...     n_encoder_layers=1,
    ...     conv_layers=[1, 1, 1],
    ...     max_epochs=1,
    ...     random_state=0,
    ... )
    >>> pipeline = AptaTransPipeline(apta_max_len=16, prot_max_len=16, estimator=clf)
    >>> pipeline.fit(X, y)  # doctest: +ELLIPSIS
    AptaTransPipeline(...)
    >>> experiment = AptamerEvalAptaTrans(protein, pipeline)
    >>> score = experiment.evaluate("AUGGC")
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
            data={
                self.pipeline.aptamer_col: [aptamer_candidate],
                self.pipeline.protein_col: [self.target],
            }
        )
        score = self.pipeline.predict_proba(X=X)

        return np.float64(score[:, 1].item())
