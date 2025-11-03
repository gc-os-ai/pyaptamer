__author__ = ["nennomp"]
__all__ = ["BaseAptamerEval"]

from abc import ABC, abstractmethod

import numpy as np
from skbase.base import BaseObject


class BaseAptamerEval(BaseObject, ABC):
    """Abstract base class for candidate aptamer evaluation against target proteins.

    This class defines the common interface and shared functionality for aptamer
    evaluation using different models or pipelines.

    Parameters
    ----------
    target : str
        Target sequence string consisting of amino acid letters and (possibly) unknown
        characters. Interpreted as the amino-acid sequence of the binding target
        protein.
    """

    def __init__(self, target: str) -> None:
        self.target = target
        super().__init__()

    def _inputnames(self) -> list[str]:
        """Return the inputs of the experiment."""
        return ["aptamer_candidate"]

    @abstractmethod
    def evaluate(self, aptamer_candidate: str, **kwargs) -> np.float64:
        """Evaluate the given aptamer candidate against the target protein.

        Parameters
        ----------
        aptamer_candidate : str
            The aptamer candidate to evaluate. It should be a string consisting of
            letters representing nucleotides: 'A', 'C', 'G', and 'U' (for RNA) or 'T'
            (for DNA).
        **kwargs
            Additional keyword arguments specific to the implementation.

        Returns
        -------
        np.float64
            The score assigned to the aptamer candidate.
        """
        pass
