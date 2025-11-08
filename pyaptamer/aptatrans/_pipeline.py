"""
AptaTrans' complete pipeline for for aptamer-protein interaction prediction and
candidate aptamers recommendation.
"""

__author__ = ["nennomp", "satvshr"]
__all__ = ["AptaTransPipeline"]

import lightning as L
import torch
from sklearn.base import BaseObject
from torch import Tensor
from torch.utils.data import DataLoader

from pyaptamer.aptatrans import AptaTrans, AptaTransLightning
from pyaptamer.datasets.dataclasses import APIDataset
from pyaptamer.experiments import AptamerEvalAptaTrans
from pyaptamer.mcts import MCTS
from pyaptamer.utils._base import filter_words

# ---- TRAINING CONSTANTS ---- #
BATCH_SIZE = 32
APTA_MAX_LEN = 100
PROT_MAX_LEN = 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50


class AptaTransPipeline(BaseObject):
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
        The depth of the tree in the Monte Carlo Tree Search (MCTS) algorithm.
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
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.depth = depth
        self.prot_words = prot_words
        self.n_iterations = n_iterations
        self.prot_words_ = None

    def _init_vocabularies(self):
        """Initialize filtered protein words vocabulary."""
        if self.prot_words_ is None:
            self.prot_words_ = filter_words(self.prot_words)

    def _init_aptamer_experiment(self, target: str) -> AptamerEvalAptaTrans:
        """Initialize the aptamer recommendation experiment."""
        self._init_vocabularies()
        experiment = AptamerEvalAptaTrans(
            target=target,
            model=self.model,
            device=self.device,
            prot_words=self.prot_words_,
        )
        return experiment

    def _preprocess_data(self, train_dataset):
        """Convert numpy arrays or pandas DataFrames into a PyTorch DataLoader."""
        self._init_vocabularies()

        dataset = APIDataset(
            x_apta=train_dataset["aptamer"].to_numpy(),
            x_prot=train_dataset["protein"].to_numpy(),
            y=train_dataset["label"].to_numpy(),
            apta_max_len=APTA_MAX_LEN,
            prot_max_len=PROT_MAX_LEN,
            prot_words=self.prot_words_,
        )

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        return dataloader

    def fit(self, train_dataset):
        """Train the AptaTrans model using PyTorch Lightning.

        This method preprocesses the training dataset, converts it into a PyTorch
        DataLoader, and trains the AptaTrans model using the AptaTransLightning
        wrapper with the PyTorch Lightning Trainer.

        Parameters
        ----------
        train_dataset : pandas.DataFrame
            A DataFrame containing the training data. Must include the following
            columns:
                - ``aptamer`` : str
                    The aptamer nucleotide sequences.
                - ``protein`` : str
                    The target protein sequences.
                - ``label`` : float
                    The ground truth interaction scores or binary labels.

        Returns
        -------
        AptaTransPipeline
            The fitted AptaTransPipeline instance with an updated and trained model.
        """
        dataloader = self._preprocess_data(train_dataset)

        model_lightning = AptaTransLightning(
            model=self.model,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
        )

        trainer.fit(model_lightning, dataloader)

        # Keep trained model
        self.model = model_lightning.model.to(self.device)

        return self

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
           A tensor of dtype torch.float64 containing the predicted interaction score.
        """
        experiment = self._init_aptamer_experiment(target)
        score = experiment.evaluate(candidate)
        return score

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
