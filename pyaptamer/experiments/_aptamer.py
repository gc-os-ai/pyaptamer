__author__ = ["nennomp"]
__all__ = ["Aptamer"]

import torch
from skbase.base import BaseObject

from pyaptamer.utils import rna2vec


class Aptamer(BaseObject):
    """Candidate aptamer evaluation for a given target protein.

    Parameters
    ----------
    target_encoded : torch.Tensor
        Encoded target sequence tensor.
    target : str, optional
        Target sequence string.
    model : torch.nn.Module
        Model to use for assigning scores.
    device : torch.device
        Device to run the model on.

    Examples
    --------
    from pyaptamer.experiment import Aptamer
    >>> experiment = Aptamer(target_encoded, target, model, device)
    >>> score = experiment.run(aptamer_candidate)
    """

    def __init__(
        self,
        target_encoded: torch.Tensor,
        target: str,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        """
        Parameters
        ----------
        target_encoded : torch.Tensor
            Encoded target sequence tensor.
        target : str, optional
            Target sequence string.
        model : torch.nn.Module
            Model to use for assigning scores.
        device : torch.device
            Device to run the model on.
        """
        super().__init__()
        self.target_encoded = target_encoded.to(device)
        self.target = target
        self.model = model
        self.device = device

    def _inputnames(self) -> list[str]:
        """Return the inputs of the experiment."""
        return ["aptamer_candidate"]

    def _reconstruct(self, sequence: str = "") -> torch.Tensor:
        """Reconstruct the actual aptamer sequence from the encoded representation.

        The encoding uses pairs like 'A_' (add A to left) and '_A' (add A to right).
        This method converts these pairs back to the actual sequence. Then, from its
        RNA sequence representation it is converted to a vector.

        Parameters
        ----------
        seq : str
            Encoded sequence with direction markers (underscores).

        Returns
        -------
        torch.Tensor
            The reconstructed RNA sequence as a vector.
        """
        result = ""
        for i in range(0, len(sequence), 2):
            match sequence[i]:
                case "_":
                    # append the next values
                    result = result + sequence[i + 1]
                case _:
                    # prepend the current value
                    result = sequence[i] + result

        return torch.tensor(rna2vec([result]))

    @torch.no_grad()
    def evaluate(self, aptamer_candidate: str) -> None:
        """Evaluate the given aptamer candidate by assigning a score.

        Parameters
        ----------
        aptamer_candidate : str
            The aptamer candidate to evaluate. It should be a string consisting of
            letters representing nucleotides: 'A_', '_A', 'C_', '_C', 'G_', '_G', 'U_',
            '_U'. Underscores indicate whether the nucleotides are supposed to be (e.
            g., 'A_') prepended or appended (e.g., '_A)'to the sequence.

        Returns
        -------
        torch.Tensor
            The score assigned to the aptamer candidate.
        """
        aptamer_candidate = self._reconstruct(aptamer_candidate)

        self.model.eval()
        return self.model(
            aptamer_candidate.to(self.device),
            self.target_encoded,
        )
