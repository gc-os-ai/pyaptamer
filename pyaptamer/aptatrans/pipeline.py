"""
AptaTrans' complete pipeline for for aptamer-protein interaction prediction and
candidate aptamers recommendation.
"""

__author__ = ["nennomp"]
__all__ = ["AptaTransPipeline"]

import numpy as np
import torch

from pyaptamer.aptatrans import AptaTrans
from pyaptamer.experiments import Aptamer
from pyaptamer.mcts import MCTS
from pyaptamer.utils import (
    encode_rna,
    generate_all_aptamer_triplets,
)


class AptaTransPipeline:
    """AptaTrans pipeline as described in [1]_.

    Original implementation:
    - https://github.com/PNUMLB/AptaTrans

    The AptaTrans pipeline combines leverages AptaTrans' deep neural network for
    aptamer-protein interaction prediction and, by combining it with Apta-MCTS [2]_,
    recommends candidate aptamers for a given target protein.

    Attributes
    ----------
    apta_words, prot_words : dict[str, int]
        A dictionary mapping aptamer and protein 3-mer subsequences to unique indices,
        respectively.

    References
    ----------
    .. [1] Shin, Incheol, et al. "AptaTrans: a deep neural network for predicting
    aptamer-protein interaction using pretrained encoders." BMC bioinformatics 24.1
    (2023): 447.
    .. [2] Lee, Gwangho, et al. "Predicting aptamer sequences that interact with target
    proteins using an aptamer-protein interaction classifier and a Monte Carlo tree
    search approach." PloS one 16.6 (2021): e0253760.

    Examples
    --------
    >>> from pyaptamer.aptatrans import AptaTrans
    >>> from pyaptamer.aptatrans import AptaTransPipeline
    >>> model = AptaTrans(apta_embedding, prot_embedding)
    >>> pipeline = AptaTransPipeline(devdevice, model, prot_words)
    >>> candidates = pipeline.recommend(target, n_candidates=3, depth=5)
    >>> print(candidates)
    {'AUGGC': 0.85, 'CAGUA': 0.78, 'GCUAG': 0.65}
    """

    def __init__(
        self,
        device: torch.device,
        model: AptaTrans,
        prot_words: dict[str, float],
    ) -> None:
        """
        Parameters
        ----------
        device : torch.device
            The device on which to run the model.
        model : AptaTrans
            An instance of the AptaTrans() class.
        apta_words, prot_words : dict[str, int]
            A dictionary mapping RNA/protein 3-mer subsequences to integer token IDs,
            respectively.
        """
        super().__init__()
        self.device = device
        self.model = model.to(device)

        self.apta_words, self.prot_words = self._init_words(prot_words)

    def _init_words(
        self, prot_words: dict[str, float]
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Initialize aptamer and protein word vocabularies.

        For aptamers, creates a mapping between all possible 3-mer RNA subsequences and
        integer indices. For proteins, 3-mers with below-average frequency are filtered
        out. Then, they are mapped to integer indices.

        Parameters
        ----------
        prot_words : dict[str, float]
            A dictionary containing protein 3-mer subsequences and their frequencies.

        Returns
        -------
        tuple[dict[str, int], dict[str, int]]
            A tuple of dictionaries mapping aptamer and protein 3-mer subsequences to
            unique indices, respectively.
        """
        # generate all possible RNA triplets (5^3 -> 125 total)
        apta_words = generate_all_aptamer_triplets()

        # filter out protein words with below average frequency
        mean_freq = np.mean(list(prot_words.values()))
        prot_words = [seq for seq, freq in prot_words.items() if freq > mean_freq]
        prot_words = {word: i + 1 for i, word in enumerate(prot_words)}

        return (apta_words, prot_words)

    def _init_aptamer_experiment(self, target: str) -> Aptamer:
        """Initialize the aptamer experiment."""
        # initialize the aptamer recommendation experiment
        target_encoded = encode_rna(
            device=self.device,
            target=target,
            words=self.prot_words,
            max_len=self.model.prot_embedding.max_len,
        )
        experiment = Aptamer(
            target_encoded=target_encoded,
            target=target,
            model=self.model,
            device=self.device,
        )
        return experiment

    def get_interaction_map(self):
        # TODO: implement this method to retrieve the intermediate output of the
        # interaction map from the neural network so that it may be used for plotting
        # TODO: ask whether this is needed/useful
        raise NotImplementedError("This method is not yet implemented.")

    def predict_api(self, candidate: str, target: str) -> torch.Tensor:
        """Predict aptamer-protein interaction (API) score for a given target protein.

        This methods initializes a new aptamer experiment for the given aptamer
        candidate and target protein. Finally, it predict the interaction score using
        the AptaTrans' deep neural network.

        Parameters
        ----------
        candidate : str
            The candidate aptamer sequence.
        target : str
            The target protein sequence.

        Returns
        -------
        torch.Tensor
            A tensor containing the predicted interaction score.
        """
        experiment = self._init_aptamer_experiment(target)
        return experiment.evaluate(candidate)

    @torch.no_grad()
    def recommend(
        self,
        target: str,
        n_candidates: int,
        depth: int = 20,
        n_iterations: int = 1000,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Recommend aptamer candidates for a given target protein.

        The Monte Carlo Tree Search (MCTS) algorithm is used to generate candidate
        aptamers. Then, AptaTrans' deep neural network is used as a scoring function to
        evaluate the candidates, inside the Aptamer() experiment.

        Parameters
        ----------
        target : str
            The target protein sequence.
        n_candidates : int
            The number of candidate aptamers to generate.
        depth : int, optional
            The depth of the tree in the MCTS algorithm.
        n_iterations : int, optional
            The number of iterations for the MCTS algorithm.
        verbose : bool, optional
            If True, enables print statements.

        Returns
        -------
        dict[str, float]
            A dictionary mapping candidates to their scores.
        """
        experiment = self._init_aptamer_experiment(target)

        # initialize MCTS with the experiment
        mcts = MCTS(
            experiment=experiment,
            depth=depth,
            n_iterations=n_iterations,
        )

        # generate aptamer candidates
        candidates = {}
        for _ in range(n_candidates):
            candidate = mcts.run()
            score = experiment.evaluate(candidate)

            if candidate not in candidates:
                candidates[candidate] = score.item()

        if verbose:
            for candidate, score in candidates.items():
                print(f"Candidate: {candidate}, Score: {score:.4f}")

        return candidates
