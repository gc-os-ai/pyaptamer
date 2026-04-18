__author__ = ["nennomp"]
__all__ = ["APIDataset"]

import numpy as np
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
    y : np.ndarray or None
        A numpy array containing labels for the interactions, where 'positive' indicates
        a positive interaction and 'negative' indicates a negative interaction.
        Pass ``None`` when using the dataset for inference only (no labels available).
    apta_max_len : int
        The maximum length for aptamer sequences after padding or truncation.
    prot_max_len : int
        The maximum length for protein sequences after padding or truncation.
    prot_words : dict[str, int]
        A dictionary mapping protein 3-mers to unique indices for encoding protein
        sequences.
    split : str, optional, default="train"
        If "train", the dataset will augment aptamer sequences by adding their
        reverse complements. If "test", the dataset will not augment the aptamer
        sequences.
        complements.
    """

    def __init__(
        self,
        x_apta: np.ndarray,
        x_prot: np.ndarray,
        y: np.ndarray | None,
        apta_max_len: int,
        prot_max_len: int,
        prot_words: dict[str, int],
        split: str = "train",
    ) -> None:
        super().__init__()

        if split not in ["train", "test", "predict"]:
            raise ValueError(
                f"Unknown split: {split}. Options are 'train', 'test', and 'predict'."
            )

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
        y: np.ndarray | None,
        split: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Prepare data by augmenting sequences and converting to numeric tensors.

        Parameters
        ----------
        x_apta : np.ndarray
            Aptamer sequences.
        x_prot : np.ndarray
            Protein sequences.
        y : np.ndarray or None
            Laabels for the interactions. None when used for inference only.
        split : str
            Dataset split. ``"train"`` augments aptamer sequences with reverse
            complements. ``"test"`` and ``"predict"`` do not augment.
        """
        if split == "train":
            x_apta = augment_reverse(x_apta)[0]
            x_prot = np.concatenate([x_prot, x_prot])
            if y is not None:
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
        if y is not None:
            y = torch.tensor((y == "positive").astype(int))

        return (x_apta, x_prot, y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.y is None:
            return (self.x_apta[index], self.x_prot[index])
        return (self.x_apta[index], self.x_prot[index], self.y[index])
