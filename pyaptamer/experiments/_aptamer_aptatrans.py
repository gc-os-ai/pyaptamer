__all__ = ["AptamerEvalAptaTrans"]

import numpy as np
import torch

from pyaptamer.experiments._aptamer import BaseAptamerEval
from pyaptamer.utils import encode_rna, rna2vec


class AptamerEvalAptaTrans(BaseAptamerEval):
    """Candidate aptamer evaluation for a given target protein using AptaTrans.

    Parameters
    ----------
    target : str
        Target sequence string consisting of amino acid letters and (possibly) unknown
        characters. Interpreted as the amino-acid sequence of the binding target
        protein.
    model : torch.nn.Module
        Model to use for assigning scores.
    device : torch.device
        Device to run the model on.
    prot_words : dict[str, float]
        A dictionary mapping protein n-mer protein subsequences to a unique integer ID.
        Used to encode protein sequences into their numerical representions.

    Attributes
    ----------
    target_encoded : Tensor
        Encoded target sequence tensor.

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig
    >>> from pyaptamer.experiments import AptamerEvalAptaTrans
    >>> apta_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> prot_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> model = AptaTrans(apta_embedding, prot_embedding)
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> target = "DHRNE"
    >>> prot_words = {"DHR": 1, "RNE": 2, "NE": 3}
    >>> experiment = AptamerEvalAptaTrans(target, model, device, prot_words)
    >>> aptamer_candidate = "AUGGC"
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
        super().__init__(target)
        self.model = model
        self.device = device

        self.target_encoded = encode_rna(
            sequences=target,
            words=prot_words,
            max_len=model.prot_embedding.max_len,
        ).to(device)

    @torch.no_grad()
    def evaluate(
        self, aptamer_candidate: str, return_interaction_map: bool = False
    ) -> np.float64 | np.ndarray:
        """Evaluate the given aptamer candidate against the target protein.

        If `return_interaction_map` is set to `True`, the method returns the
        aptamer-protein interaction map. Otherwise, it returns the score assigned to
        the aptamer candidate.

        Parameters
        ----------
        aptamer_candidate : str
            The aptamer candidate to evaluate. It should be a string consisting of
            letters representing nucleotides: 'A', 'C', 'G', and 'U' (for RNA) or 'T'
            (for DNA).
        return_interaction_map : bool, optional, default=False
            Whether to return the interaction map or not.

        Returns
        -------
        np.float64 or np.ndarray
            The score assigned to the aptamer candidate if `return_interaction_map` is
            `False`. If `return_interaction_map` is `True`, the interaction map, of
            shape (batch_size, 1, seq_len_aptamer, seq_len_protein).
        """
        self.model.eval()

        # convert the aptamer candidate to its numerical representation
        aptamer_candidate = torch.tensor(
            rna2vec(
                [aptamer_candidate],
                max_sequence_length=self.model.apta_embedding.max_len,
            ),
            dtype=torch.int64,
        )

        if return_interaction_map:
            return (
                self.model.forward_imap(
                    aptamer_candidate.to(self.device),
                    self.target_encoded,
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            return np.float64(
                self.model(
                    aptamer_candidate.to(self.device),
                    self.target_encoded,
                ).item()
            )
