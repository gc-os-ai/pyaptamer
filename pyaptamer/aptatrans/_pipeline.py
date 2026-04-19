"""
AptaTrans' complete pipeline for for aptamer-protein interaction prediction and
candidate aptamers recommendation.
"""

__author__ = ["nennomp"]
__all__ = ["AptaTransPipeline"]

import numpy as np
import torch
from torch import Tensor

from pyaptamer.aptatrans import AptaTrans
from pyaptamer.aptatrans._plotting import _render_interaction_map
from pyaptamer.experiments import AptamerEvalAptaTrans
from pyaptamer.mcts import MCTS
from pyaptamer.utils import encode_rna, generate_nplets, rna2vec
from pyaptamer.utils._base import filter_words


class AptaTransPipeline:
    """AptaTrans pipeline for aptamer affinity prediction, by Shin et al.

    Algorithm as originally described in Shin et al [1]_.

    Reimplemented based on the publication and the original codebase.

    Original implementation: https://github.com/PNUMLB/AptaTrans.

    The AptaTrans pipeline combines leverages AptaTrans' deep neural network for
    aptamer-protein interaction prediction and, by combining it with Apta-MCTS [2]_,
    recommends candidate aptamers for a given target protein.

    Parameters
    ----------
    device : torch.device
        The device on which to run the model.
    model : AptaTrans
        An instance of the AptaTrans() class.
    prot_words : dict[str, float]
        A dictionary mapping protein n-mer protein subsequences to a unique integer ID.
        Used to encode protein sequences into their numerical representions. The
        subsequences and their frequency should come from the same dataset used for
        pretraining the protein encoder.
    depth : int, optional, default=20
        The depth of the tree in the Monte Carlo Tree Search (MCTS) algorithm. Also
        defines the length of the generated aptamer candidates. Must be equal or
        greater than 3 since preprocessing uses triplet encoding (3-mers), which
        requires sequences of at least 3 nucleotides to extract overlapping triplets.
    n_iterations : int, optional, default=1000
        The number of iterations for the MCTS algorithm.

    Attributes
    ----------
    apta_words, prot_words : dict[str, int]
        A dictionary mapping aptamer 3-mer subsequences to unique indices, and protein
        words to their frequency. In particular, `prot_words` now contains only protein
        words with above-average frequency, mapped to unique integer IDs

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
    >>> import torch
    >>> from pyaptamer.aptatrans import (
    ...     AptaTrans,
    ...     AptaTransPipeline,
    ...     EncoderPredictorConfig,
    ... )
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> apta_embedding = EncoderPredictorConfig(128, 16, max_len=100)
    >>> prot_embedding = EncoderPredictorConfig(128, 16, max_len=100)
    >>> prot_words = {"DHR": 0.5, "AIQ": 0.5, "AAG": 0.2}
    >>> target = "DHRNENIAIQ"
    >>> model = AptaTrans(apta_embedding, prot_embedding)
    >>> pipeline = AptaTransPipeline(device, model, prot_words, depth=5, n_iterations=5)
    >>> aptamer = "ACGUA"
    >>> imap = pipeline.get_interaction_map(aptamer, target)
    >>> candidates = pipeline.recommend(target, n_candidates=1, verbose=False)
    """

    def __init__(
        self,
        device: torch.device,
        model: AptaTrans,
        prot_words: dict[str, float],
        depth: int = 20,
        n_iterations: int = 1000,
    ) -> None:
        """
        Raises
        ------
        ValueError
            If `depth` is less than 3.
        """
        if depth < 3:
            raise ValueError(
                f"Invalid depth value: {depth}. Must be grater or equal than 3."
            )

        self.device = device
        self.model = model.to(device)
        self.depth = depth
        self.n_iterations = n_iterations

        self.apta_words, self.prot_words = self._init_words(prot_words)

    def _init_words(
        self,
        prot_words: dict[str, float],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Initialize aptamer and protein word vocabularies.

        For aptamers, creates a mapping between all possible 3-mer RNA subsequences and
        integer indices. For proteins, load protein words mapped to their frequency,
        filter out those with below-average frequency, and assign unique integer IDs.

        Parameters
        ----------
        prot_words : dict[str, float]
            A dictionary containing protein 3-mer subsequences and their frequencies.

        Returns
        -------
        tuple[dict[str, int], dict[str, int]]
            A tuple of dictionaries mapping aptamer 3-mer subsequences to unique
            indices and protein words to their frequencies, respectively.
        """
        # generate all possible RNA triplets (5^3 -> 125 total)
        apta_words = generate_nplets(letters=["A", "C", "G", "U", "N"], repeat=3)

        # filter out protein words with below average frequency and assign unique
        # integer IDs
        prot_words = filter_words(prot_words)

        return (apta_words, prot_words)

    def _init_aptamer_experiment(self, target: str) -> AptamerEvalAptaTrans:
        """Initialize the aptamer recommendation experiment."""
        experiment = AptamerEvalAptaTrans(
            target=target,
            model=self.model,
            device=self.device,
            prot_words=self.prot_words,
        )
        return experiment

    def get_interaction_map(self, candidate: str, target: str) -> Tensor:
        # TODO: to make the interaction map ready for plotting (at least if we were to
        # follow the original paper), there are additional steps. Need to decide if put
        # it here or elsewhere (e.g., in plotting code). For now, TBD.
        # Personally, I would leave this as is, to provide an "untouched" interaction
        # map.
        """Generate the aptamer-protein interaction map.

        Parameters
        ----------
        candidate : str
            The candidate aptamer sequence.
        target : str
            The target protein sequence.

        Returns
        -------
        Tensor
            A tensor containing the interaction map, of shape (batch_size, 1,
            seq_len_apta, seq_len_prot).
        """
        experiment = self._init_aptamer_experiment(target)
        return experiment.evaluate(candidate, return_interaction_map=True)

    @torch.no_grad()
    def plot_interaction_map(
        self,
        candidate: str,
        target: str,
        view: str | None = None,
        top_k: int = 10,
        ax=None,
        figsize: tuple[int, int] = (20, 8),
    ):
        """Plot the aptamer-protein interaction map as a heatmap.

        Refactored from the original AptaTrans authors' implementation [1]_.
        Tokenizes both sequences, trims the raw interaction map to the actual
        (non-padded) token lengths, applies a softmax view, and renders a heatmap
        with the top-k most interacting token pairs labelled on the axes.

        Parameters
        ----------
        candidate : str
            The candidate aptamer sequence.
        target : str
            The target protein sequence.
        view : str or None, optional, default=None
            Which axis to highlight top interactions on. One of:

            * ``"apta"`` / ``"aptamer"`` — softmax over aptamer axis; highlights
              top-k aptamer tokens with highest cumulative interaction.
            * ``"prot"`` / ``"protein"`` / ``"target"`` — softmax over protein axis;
              highlights top-k protein tokens.
            * ``None`` — combined view; applies both softmaxes and highlights top-k
              tokens on both axes.
        top_k : int, optional, default=10
            Number of top interacting tokens to label on each highlighted axis.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure and axes are created.
        figsize : tuple[int, int], optional, default=(20, 8)
            Figure size when creating a new figure (ignored if ``ax`` is given).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the rendered heatmap.

        References
        ----------
        .. [1] Original AptaTrans pipeline:
               https://github.com/PNUMLB/AptaTrans/blob/master/aptatrans_pipeline.py
        """
        # tokenize aptamer: rna2vec produces overlapping 3-mers
        apta_tokenized = torch.tensor(
            rna2vec([candidate], max_sequence_length=self.model.apta_embedding.max_len),
            dtype=torch.int64,
        )  # shape: (1, max_apta_len)

        # tokenize protein: greedy longest-match against prot_words vocabulary
        prot_tokenized = encode_rna(
            sequences=target,
            words=self.prot_words,
            max_len=self.model.prot_embedding.max_len,
        )  # shape: (1, max_prot_len)

        # find actual (non-padded) token counts
        apta_nonpad = apta_tokenized[0][apta_tokenized[0] != 0]
        prot_nonpad = prot_tokenized[0][prot_tokenized[0] != 0]
        n_apta = len(apta_nonpad)
        n_prot = len(prot_nonpad)

        # decode token indices back to 3-mer strings
        reversed_apta_words = {v: k for k, v in self.apta_words.items()}
        reversed_prot_words = {v: k for k, v in self.prot_words.items()}
        apta_tokens = [reversed_apta_words.get(idx.item(), "?") for idx in apta_nonpad]
        prot_tokens = [reversed_prot_words.get(idx.item(), "?") for idx in prot_nonpad]

        # get raw interaction map and trim to actual sequence lengths
        im_raw = self.get_interaction_map(candidate, target)  # numpy (1,1,H,W)
        im = torch.tensor(im_raw)[0, 0, :n_apta, :n_prot]  # (n_apta, n_prot)

        # apply softmax and compute top-k indices per view
        top_k = min(top_k, n_apta, n_prot)
        if view in ("apta", "aptamer"):
            scores = im.softmax(dim=0).sum(dim=1)
            apta_indices = torch.argsort(scores)[-top_k:].tolist()
            prot_indices = list(range(n_prot))
        elif view in ("prot", "protein", "target"):
            scores = im.softmax(dim=1).sum(dim=0)
            prot_indices = torch.argsort(scores)[-top_k:].tolist()
            apta_indices = list(range(n_apta))
        else:  # combined: highlight top-k on both axes
            apta_scores = im.softmax(dim=0).sum(dim=1)
            prot_scores = im.softmax(dim=1).sum(dim=0)
            apta_indices = torch.argsort(apta_scores)[-top_k:].tolist()
            prot_indices = torch.argsort(prot_scores)[-top_k:].tolist()

        im_np = im.cpu().numpy()
        return _render_interaction_map(
            im_np, apta_tokens, prot_tokens, apta_indices, prot_indices, ax, figsize
        )

    def predict(self, candidate: str, target: str) -> Tensor:
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
        Tensor
            A tensor containing the predicted interaction score.
        """
        experiment = self._init_aptamer_experiment(target)
        return experiment.evaluate(candidate)

    @torch.no_grad()
    def recommend(
        self,
        target: str,
        n_candidates: int = 10,
        verbose: bool = True,
    ) -> set[tuple[str, str, float]]:
        """Recommend aptamer candidates for a given target protein.

        The Monte Carlo Tree Search (MCTS) algorithm is used to generate candidate
        aptamers. Then, AptaTrans' deep neural network is used as a scoring function to
        evaluate the candidates, inside the Aptamer() experiment. The process stop when
        `n_candidates` unique candidates are generated.

        Parameters
        ----------
        target : str
            The target protein sequence.
        n_candidates : int, optional, default=10
            The number of candidate aptamers to generate.
        verbose : bool, optional, default=True
            If True, enables print statements for debugging and progress tracking.

        Returns
        -------
        set[tuple[str, str, float]]
            A set of tuples containing reconstructed and unrecontructed candidate
            aptamer sequence, and the corresponding score.
        """
        experiment = self._init_aptamer_experiment(target)

        # initialize MCTS with the experiment
        mcts = MCTS(
            experiment=experiment,
            depth=self.depth,
            n_iterations=self.n_iterations,
        )

        # generate aptamer candidates
        candidates = set()
        while len(candidates) < n_candidates:
            candidate = mcts.run(verbose=verbose)
            candidates.add(tuple(candidate.values()))

        if verbose:
            for candidate, sequence, score in candidates:
                print(
                    f"Candidate: {candidate}, "
                    f"Sequence: {sequence}, "
                    f"Score: {score.item():.4f}"
                )

        return candidates
