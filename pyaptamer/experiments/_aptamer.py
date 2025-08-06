__author__ = ["nennomp"]
__all__ = ["Aptamer"]

import torch
from skbase.base import BaseObject
from torch import Tensor

from pyaptamer.utils import encode_rna, rna2vec


class Aptamer(BaseObject):
    """Candidate aptamer evaluation for a given target protein.

    Parameters
    ----------
    target : str, optional
        Target sequence string.
    model : torch.nn.Module
        Model to use for assigning scores.
    device : torch.device
        Device to run the model on.
    prot_words : dict[str, int]
        A dictionary mapping protein 3-mer subsequences to integer token IDs.

    Attributes
    ----------
    target_encoded : Tensor
        Encoded target sequence tensor.

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.aptatrans import AptaTrans
    >>> from pyaptamer.experiment import Aptamer
    >>> target = "DHRNE"
    >>> aptamer_candidate = "AUGGC"
    >>> model = AptaTrans(apta_embedding, prot_embedding)
    >>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device
    ... ("cpu")
    >>> prot_words = {"AAA": 0.5, "AAC": 0.3, "AAG": 0.2, ...}
    >>> experiment = Aptamer(target, model, device, prot_words)
    >>> imap = experiment.evaluate(aptamer_candidate, return_interaction_map=True)
    >>> score = experiment.evaluate(aptamer_candidate)
    """

    def __init__(
        self,
        target: str,
        model: torch.nn.Module,
        device: torch.device,
        prot_words: dict[str, int],
    ) -> None:
        self.target = target
        self.model = model
        self.device = device

        self.target_encoded = encode_rna(
            sequences=target,
            words=prot_words,
            max_len=model.prot_embedding.max_len,
        ).to(device)

        super().__init__()

    def _inputnames(self) -> list[str]:
        """Return the inputs of the experiment."""
        return ["aptamer_candidate"]

    def _reconstruct(self, sequence: str = "") -> Tensor:
        """Reconstruct the actual aptamer sequence from the encoded representation.

        The expected encoded representation uses pairs like 'A_' (add A to left) and
        '_A' (add A to right). This method converts these pairs back to the actual
        sequence, then converts the RNA sequence representation to a vector. If no
        underscores are present, the sequence is assumed to be already reconstructed
        and simply converted to its vector representation.

        Parameters
        ----------
        sequence : str, optional, default=""
            Encoded sequence with direction markers (underscores) or an already
            reconstructed sequence (without underscores).

        Returns
        -------
        Tensor
            The reconstructed RNA sequence as a vector of shape (1, seq_len), depending
            on rna2vec's `max_sequence_length` parameter.

        Raises
        -------
        AssertionError
            If the encoded sequence has an odd length, indicating it is not properly
            formatted.
        """
        # already reconstructed
        if "_" not in sequence:
            return Tensor(rna2vec([sequence]))

        # if the sequence is not reconstructed yet, it should have an even length
        # because it should consist of pairs such as 'A_' and '_A' (i.e., nucleotide +
        # direction marker).
        assert len(sequence) % 2 == 0, (
            f"Encoded sequence must have even length, got {len(sequence)}."
        )

        # reconstruct
        result = ""
        for i in range(0, len(sequence), 2):
            if sequence[i] == "_":
                # append to right
                result += sequence[i + 1]
            else:
                # prepend to left
                result = sequence[i] + result

        return Tensor(rna2vec([result]))

    @torch.no_grad()
    def evaluate(
        self, aptamer_candidate: str, return_interaction_map: bool = False
    ) -> Tensor:
        """Evaluate the given aptamer candidate against the target protein.

        If `return_interaction_map` is set to `True`, the method returns the
        aptamer-protein interaction map. Otherwise, it returns the score assigned to
        the aptamer candidate.

        Parameters
        ----------
        aptamer_candidate : str
            The aptamer candidate to evaluate. It should be a string consisting of
            letters representing nucleotides: 'A_', '_A', 'C_', '_C', 'G_', '_G', 'U_',
            '_U'. Underscores indicate whether the nucleotides are supposed to be (e.
            g., 'A_') prepended or appended (e.g., '_A)'to the sequence.
        return_interaction_map : bool, optional, default=False
            Whether to return the interaction map or not.

        Returns
        -------
        Tensor
            The score assigned to the aptamer candidate if `return_interaction_map` is
            `False`. If `return_interaction_map` is `True`, the interaction map, of
            shape (batch_size, 1, seq_len_aptamer, seq_len_protein).
        """
        aptamer_candidate = self._reconstruct(aptamer_candidate)
        self.model.eval()

        if return_interaction_map:
            return self.model.forward_imap(
                aptamer_candidate.to(self.device),
                self.target_encoded,
            )
        else:
            return self.model(
                aptamer_candidate.to(self.device),
                self.target_encoded,
            )
