__author__ = ["nennomp"]
__all__ = ["APIDataset"]

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pyaptamer.utils import encode_rna, rna2vec
from pyaptamer.utils._augment import augment_reverse


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
    split : str, optional, default="train"
        If "train", the dataset will augment aptamer sequences by adding their
        reversed sequences. If "test", the dataset will not augment the aptamer
        sequences.
    """

    def __init__(
        self,
        x_apta: np.ndarray,
        x_prot: np.ndarray,
        y: np.ndarray,
        apta_max_len: int,
        prot_max_len: int,
        prot_words: dict[str, int],
        split: str = "train",
    ) -> None:
        super().__init__()

        if split not in ["train", "test"]:
            raise ValueError(f"Unknown split: {split}. Options are 'train' and 'test'.")

        self.apta_max_len = apta_max_len
        self.prot_max_len = prot_max_len
        self.prot_words = prot_words
        self.split = split

        self.x_apta, self.x_prot, self.y = self._prepare_data(x_apta, x_prot, y, split)

        self.len = len(self.x_apta)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        apta_col: str,
        prot_col: str,
        label_col: str,
        apta_max_len: int,
        prot_max_len: int,
        prot_words: dict[str, int],
        split: str = "train",
    ) -> "APIDataset":
        """
        Create an APIDataset from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data.
        apta_col : str
            The name of the column containing aptamer sequences.
        prot_col : str
            The name of the column containing protein sequences.
        label_col : str
            The name of the column containing interaction labels.
        apta_max_len : int
            Maximum length for aptamer sequences.
        prot_max_len : int
            Maximum length for protein sequences.
        prot_words : dict[str, int]
            Protein k-mer word mapping.
        split : str, optional, default="train"
            If "train", the dataset will augment aptamer sequences.

        Returns
        -------
        APIDataset
            An instance of APIDataset.
        """
        x_apta = df[apta_col].values
        x_prot = df[prot_col].values
        y = df[label_col].values
        return cls(
            x_apta=x_apta,
            x_prot=x_prot,
            y=y,
            apta_max_len=apta_max_len,
            prot_max_len=prot_max_len,
            prot_words=prot_words,
            split=split,
        )

    def _prepare_data(
        self,
        x_apta: np.ndarray,
        x_prot: np.ndarray,
        y: np.ndarray,
        split: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the data by augmenting aptamer sequences with their reversed sequences
        and transforming them to vector numerical representations.

        Parameters
        ----------
        x_apta : np.ndarray
            Aptamer sequences.
        x_prot : np.ndarray
            Protein sequences.
        y : np.ndarray
            Laabels for the interactions.
        split : str
            If "train", the dataset will augment aptamer sequences by adding
            their reversed sequences.
        """
        if split == "train":
            x_apta = augment_reverse(x_apta)[0]
            x_prot = np.concatenate([x_prot, x_prot])
            y = np.concatenate([y, y])

        x_apta = torch.tensor(
            rna2vec(
                sequence_list=x_apta,
                max_sequence_length=self.apta_max_len,
                sequence_type="rna",
            )
        )
        x_prot = encode_rna(
            sequences=x_prot,
            words=self.prot_words,
            max_len=self.prot_max_len,
        )
        y = torch.tensor((y == "positive").astype(int))

        return (x_apta, x_prot, y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.x_apta[index], self.x_prot[index], self.y[index])
