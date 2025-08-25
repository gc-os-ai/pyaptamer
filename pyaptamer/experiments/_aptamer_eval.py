__author__ = ["nennomp"]
__all__ = ["AptamerEvalAptaNet", "AptamerEvalAptaTrans"]

from abc import ABC, abstractmethod
import torch
from skbase.base import BaseObject
from torch import Tensor

from pyaptamer.utils import encode_rna, rna2vec


class BaseAptamerEval(BaseObject, ABC):
    """Abstract base class for candidate aptamer evaluation against target proteins.

    This class defines the common interface and shared functionality for aptamer
    evaluation using different models or pipelines.

    Parameters
    ----------
    target : str
        Target sequence string.

    Attributes
    ----------
    target : str
        Target sequence string.
    """

    def __init__(self, target: str) -> None:
        self.target = target
        super().__init__()

    def _inputnames(self) -> list[str]:
        """Return the inputs of the experiment."""
        return ["aptamer_candidate"]

    def reconstruct(self, sequence: str = "") -> str:
        """Reconstruct the actual aptamer sequence from the encoded representation.

        The encoding uses pairs like 'A_' (add A to left) and '_A' (add A to right).
        This method converts these pairs back to the actual sequence.

        Parameters
        ----------
        sequence : str
            Encoded sequence with direction markers (underscores).

        Returns
        -------
        str
            The reconstructed RNA sequence.
        """
        # already reconstructed
        if "_" not in sequence:
            return sequence

        # if the sequence is not reconstructed yet, it should have an even length
        # because it should consist of pairs such as 'A_' and '_A' (i.e., nucleotide +
        # direction marker).
        assert len(sequence) % 2 == 0, (
            f"Encoded sequence must have even length, got {len(sequence)}."
        )

        # reconstruct
        result = ""
        for i in range(0, len(sequence), 2):
            match sequence[i]:
                case "_":
                    # append the next values
                    result = result + sequence[i + 1]
                case _:
                    # prepend the current value
                    result = sequence[i] + result

        return result

    @abstractmethod
    def evaluate(self, aptamer_candidate: str, **kwargs) -> Tensor:
        """Evaluate the given aptamer candidate against the target protein.

        Parameters
        ----------
        aptamer_candidate : str
            The aptamer candidate to evaluate.
        **kwargs
            Additional keyword arguments specific to the implementation.

        Returns
        -------
        Tensor
            The score assigned to the aptamer candidate.
        """
        pass


class AptamerEvalAptaTrans(BaseAptamerEval):
    """Candidate aptamer evaluation for a given target protein using AptaTrans.

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
    >>> from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig
    >>> from pyaptamer.experiments import AptamerEvalAptaTrans
    >>> apta_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> prot_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> model = AptaTrans(apta_embedding, prot_embedding)
    >>> target = "DHRNE"
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> prot_words = {"AAA": 0.5, "AAC": 0.3, "AAG": 0.2}
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

    def reconstruct(self, sequence: str = "") -> tuple[str, Tensor]:
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
        tuple[str, torch.Tensor]
            The reconstructed RNA sequence and its vector representation.
        """
        # get the base reconstructed sequence
        reconstructed_seq = super().reconstruct(sequence)
        
        # already reconstructed case
        if "_" not in sequence:
            return (
                reconstructed_seq,
                torch.tensor(
                    rna2vec(
                        [reconstructed_seq],
                        max_sequence_length=self.model.apta_embedding.max_len,
                    ),
                    dtype=torch.int64,
                ),
            )

        # return tuple with tensor representation
        return (
            reconstructed_seq,
            torch.tensor(
                rna2vec(
                    [reconstructed_seq], max_sequence_length=self.model.apta_embedding.max_len
                ),
                dtype=torch.int64,
            ),
        )

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
        aptamer_candidate = self.reconstruct(aptamer_candidate)[1]
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


class AptamerEvalAptaNet(BaseAptamerEval):
    """Candidate aptamer evaluation for a given target protein using AptaNet.

    Parameters
    ----------
    target : str
        Target protein sequence string.
    pipeline : AptaNetPipeline
        Fitted AptaNetPipeline to use for assigning scores.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptanet import AptaNetPipeline
    >>> from pyaptamer.experiments import AptamerEvalAptaNet
    >>> aptamer_seq = 'AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC'
    >>> protein_seq = 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY'
    >>> pairs = [(aptamer_seq, protein_seq) for _ in range(5)]
    >>> labels = np.array([0] * 5, dtype=np.float32)
    >>> pipeline = AptaNetPipeline()
    >>> pipeline.fit(pairs, labels)
    >>> target = "ACDEFACDEFACDEFACDEFACDEFACDEFACDEFACDEF"
    >>> experiment = AptamerEvalAptaNet(target, pipeline)
    >>> aptamer_candidate = "A_U_G_G_C_"
    >>> score = experiment.evaluate(aptamer_candidate)
    """

    def __init__(self, target: str, pipeline) -> None:
        super().__init__(target)
        self.pipeline = pipeline

    def evaluate(self, aptamer_candidate: str) -> Tensor:
        """Evaluate the given aptamer candidate against the target protein.

        Parameters
        ----------
        aptamer_candidate : str
            The aptamer candidate to evaluate. It should be a string consisting of
            letters representing nucleotides: 'A_', '_A', 'C_', '_C', 'G_', '_G', 'U_',
            '_U'. Underscores indicate whether the nucleotides are supposed to be (e.
            g., 'A_') prepended or appended (e.g., '_A') to the sequence.

        Returns
        -------
        Tensor
            The probability score assigned to the aptamer candidate.
        """
        aptamer_seq = self.reconstruct(aptamer_candidate)
        score = self.pipeline.predict(
            X=[(aptamer_seq, self.target)], 
            output_type="proba"
        )

        return torch.tensor(score)