__author__ = ["nennomp"]
__all__ = ["APIDataset"]

import numpy as np
import torch
from torch.utils.data import Dataset

from pyaptamer.utils._protein import encode_protein
from pyaptamer.utils._rna import rna2vec
from pyaptamer.utils.augment import augment_reverse


class APIDataset(Dataset):
    """A PyTorch dataset for aptamer-protein interaction (API) data.

    Parameters
    ----------
    x_apta : np.ndarray
        A numpy array containing aptamer sequences.
    x_prot : np.ndarray
        A numpy array containing protein sequences.
    y : np.ndarray
        A numpy array containing labels for the interactions, where 'positive' indicates
        a positive interaction and 'negative' indicates a negative interaction.
    apta_max_len : int
        The maximum length for aptamer sequences after padding or truncation.
    prot_max_len : int
        The maximum length for protein sequences after padding or truncation.
    prot_words : dict[str, int]
        A dictionary mapping protein 3-mers to unique indices for encoding protein
        sequences.
    split : bool, optional, default=True
        If True, the dataset will augment aptamer sequences by adding their reverse
        complements.
    """

    def __init__(
        self,
        x_apta: np.ndarray,
        x_prot: np.ndarray,
        y: np.ndarray,
        apta_max_len: int,
        prot_max_len: int,
        prot_words: dict[str, int],
        split: bool = True,
    ) -> None:
        super().__init__()

        self.apta_max_len = apta_max_len
        self.prot_max_len = prot_max_len
        self.prot_words = prot_words
        self.split = split

        self.x_apta, self.x_prot, self.y = self._prepare_data(x_apta, x_prot, y, split)

        self.len = len(self.x_apta)

    def _prepare_data(
        self,
        x_apta: np.ndarray,
        x_prot: np.ndarray,
        y: np.ndarray,
        split: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the data by augmenting aptamer sequences with their reverse complements
        and transforming them to vector numericla representations.
        """

        if split == "train":
            x_apta = augment_reverse(x_apta)[0]
            x_prot = np.concatenate([x_prot, x_prot])
            y = np.concatenate([y, y])

        x_apta = (
            rna2vec(
                sequence_list=x_apta,
                max_sequence_length=self.apta_max_len,
                sequence_type="rna",
            ),
        )
        x_prot = encode_protein(
            sequences=x_prot,
            words=self.prot_words,
            max_len=self.prot_max_len,
        )
        y = (y == "positive").astype(int)

        return (
            torch.tensor(x_apta, dtype=torch.int64),
            torch.tensor(x_prot, dtype=torch.int64),
            torch.tensor(y, dtype=torch.int64),
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_apta[index], self.x_prot[index], self.y[index]
