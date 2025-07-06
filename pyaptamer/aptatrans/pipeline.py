"""Implementation of AptaTrans pipeline [1][2].

This module implements the entire AptaTrans pipeline, combining AptaTrans' deep neural network and Apta-MCTS [3][4], for predicting aptamer-protein interactions and generating candidate aptamers 
for a given target protein.

References:
[1] Shin, Incheol, et al. "AptaTrans: a deep neural network for predicting aptamer-protein 
interaction using pretrained encoders." BMC bioinformatics (2023)
[2] https://github.com/PNUMLB/AptaTrans
[3] Lee, Gwangho, et al. "Predicting aptamer sequences that interact with target proteins using an 
aptamer-protein interaction classifier and a Monte Carlo tree search approach." PloS one (2021)
[4] https://github.com/leekh7411/Apta-MCTS
"""

__author__ = ["nennomp"]
__all__ = ["AptaTransPipeline"]

from itertools import product

import numpy as np
import torch
from torch import Tensor

from pyaptamer.mcts.algorithm import MCTS
from pyaptamer.aptatrans.model import AptaTrans
from pyaptamer.aptatrans.utils import encode_protein, rna2vec


class AptaTransPipeline:
    """
    
    Attributes:
        apta_words: A dictionary mapping RNA aptamer 3-mer subsequences to integer token IDs for 
            neural network input. Contains all possible 3-character combinations of RNA 
            nucleotides (A, C, G, U, N).
        prot_words: A Dictionary mapping protein 3-mer subsequences to integer token IDs. Contains 
            only above-average frequency subsequences from the training dataset.
    """
    def __init__(
        self,
        device: torch.device,
        model: 'AptaTrans',
        prot_words: dict[str, float],
    ) -> None:
        """
        Initialize the AptaTransPipeline.
        
        Args:
            model: An instance of the AptaTrans() class.
            prot_words: A dictionary mapping protein 3-mer subsequences to their frequencies in 
                the training dataset. Keys are 3-mer sequences and values are their frequencies.

        Raises:
            ValueError: If model is not an instance of AptaTrans() class.
        """
        super().__init__()
        if not isinstance(model, AptaTrans):
            raise ValueError(f"'model' must be an instance of AptaTrans() class.")
        
        self.device = device
        self.model = model.to(device)

        self.apta_words, self.prot_words = self._init_words(prot_words)

    def _init_words(self, prot_words: dict[str, float]) -> tuple[dict[str, int], dict[str, int]]:
        """Initialize aptamer and protein word vocabularies.
        
        For aptamers, creates a mapping between all possible 3-mer RNA subsequences and 
        integer indices. For proteins, 3-mers with below-average frequency are filtered out. Then, 
        they are mapped to integer indices.
        
        Args:
            prot_words: A dictionary containing protein 3-mer subsequences and their frequencies 
                from the training dataset.
        
        Returns:
            A tuple containing two dictionaries mapping RNA 3-mers and protein 3-mers to integer 
            token IDs, respectively.
        """
        # Generate all possible 3-character RNA combinations (5^3 -> 125 total)
        letters = ['A', 'C', 'G', 'U', 'N']
        apta_words = {
            ''.join(triplet): i + 1 
            for i, triplet in enumerate(product(letters, repeat=3))
        }

        # Filter out protein words with below-average frequency
        mean_freq = np.mean(list(prot_words.values()))
        prot_words = [seq for seq, freq in prot_words.items() if freq > mean_freq]
        prot_words = {word: i + 1 for i, word in enumerate(prot_words)}

        return (apta_words, prot_words)

    @torch.no_grad()
    def _evaluate_candidate(
        self, 
        candidate: str,
        target: Tensor,
    ) -> float:
        """Evaluate a given aptamer candidate for a given target protein.
        
        Args:
            mcts: An instance of the MCTS() class.
            target: (encoded) Target protein.

        Returns:
            A score related to the given candidate.
        """
        self.model.eval()

        candidate = torch.tensor(rna2vec(np.array([candidate])), dtype=torch.int64).to(self.device)
        score = self.model(candidate, target)

        return score

    @torch.no_grad()
    def recommend(
        self, 
        target: str, 
        n_candidates: int, 
        depth: int, 
        n_iterations: int, 
        verbose: bool = True
    ) -> dict[str, float]:
        """Recommend aptamer candidates for a given target protein.

        Args:
            target: A target protein sequence.
            n_candidates: Number of aptamer candidates to generate.
            depth: Maximum depth for the MCTS algorithm.
            n_iterations: Number of iterations for the MCTS algorithm.

        Returns:
            A dictionary containing mapping candidates to their scores.
        """
        encoded_target = encode_protein(
            device=self.device, 
            target=target, 
            words=self.prot_words, 
            max_len=self.model.prot_embedding.max_len,
        )
        mcts = MCTS(
            device=self.device,
            target_encoded=encoded_target,
            target=target,
            depth=depth,
            n_iterations=n_iterations,
        )

        candidates = {}
        for _ in range(n_candidates):
            mcts.make_candidate(self.model)
            candidate = mcts.get_candidate()
            score = self._evaluate_candidate(candidate, encoded_target)
            # TODO: need to check this when we refactor MCTS, to make sure it doesn't break the 
            # algorithm
            mcts.reset()

            if candidate not in candidates:
                candidates[candidate] = score.item()

        if verbose:
            for candidate, score in candidates.items():
                print(f'Candidate: {candidate}, Score: {score:.4f}')

        return candidates
