__author__ = ["nennomp"]
__all__ = ["MaskedDataset"]

import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MaskedDataset(Dataset):
    """A PyTorch dataset for masked language modeling on DNA/RNA sequences.

    Original implementation: https://github.com/PNUMLB/AptaTrans/blob/master/utils.py

    This dataset implements random masking for sequence data, where a portion
    of tokens are masked and need to be predicted. For RNA sequences, it also
    masks adjacent nucleotides to account for base pairing.

    Parameters
    ----------
    x, y : list | np.ndarray
        Numerical arrays representing input sequences and target sequences,
        respectively. Each sequence should be encoded as integers where 0 represents
        padding or unknown tokens.
    max_len : int
        Maximum sequence length used to create position indices.
    mask_idx : int
        Token index used to replace masked positions.
    masked_rate : float, optional, default=0.15
        Proportion of non-padding tokens to mask (should be between 0.0 and 1.0).
    is_rna : bool, optional, default=False
        Whether the sequences are RNA (True) or DNA (False). For RNA sequences,
        adjacent nucleotides are also masked to account for base pairing.
        Default is False.

    Attributes
    ----------
    box : np.ndarray
        Array of indices from 0 to `max_len - 1` used for masking.
    len : int
        Number of sequences in the dataset.

    Raises
    ------
    ValueError
        If the lengths of `x` and `y` do not match.

    Examples
    --------
    >>> from pyaptamer.datasets.dataclasses import MaskedDataset
    >>> sequences = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    >>> targets = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    >>> dataset = MaskedDataset(
    ...     sequences, targets, max_len=5, mask_idx=5, masked_rate=0.2, is_rna=True
    ... )
    >>> len(dataset)
    2
    """

    def __init__(
        self,
        x: list | np.ndarray,
        y: list | np.ndarray,
        max_len: int,
        mask_idx: int,
        masked_rate: float = 0.15,
        is_rna: bool = False,
    ) -> None:
        super().__init__()

        if len(x) != len(y):
            raise ValueError(
                f"Input and target arrays must have the same length. "
                f"Got x: {len(x)}, y: {len(y)}"
            )

        self.x, y = np.array(x), np.array(y)
        self.max_len = max_len
        self.mask_idx = mask_idx
        self.masked_rate = masked_rate
        self.is_rna = is_rna

        self.box = np.array(list(range(max_len)))
        self.len = len(self.x)

    def _mask_rna(self, x_masked: Tensor, mask_positions: list[int]) -> Tensor:
        """Mask adjacent nucleotides for RNA sequences.

        Parameters
        ----------
        x_masked : Tensor
            The tensor containing the masked sequence.
        mask_positions : list[int]
            List of positions that have been masked.

        Returns
        -------
        Tensor
            The tensor with adjacent nucleotides masked.
        """
        adjacent_positions = []
        for pos in mask_positions:
            # mask position + 1 (if within bounds)
            if pos < self.max_len - 1:
                adjacent_positions.append(pos + 1)
            # mask position - 1 (if within bounds)
            if pos > 0:
                adjacent_positions.append(pos - 1)
        x_masked[adjacent_positions] = self.mask_idx

        return x_masked

    def __len__(self) -> int:
        """
        Return the number of sequences in the dataset.

        Returns
        -------
        int
            Number of sequences in the dataset.
        """
        return self.len

    # TODO: For now this method applies masking as originally intended in AptaTrans
    # code. However, there may some errors:
    # (1.) 80% of the positions are masked but the remaining 20% are not masked at all.
    # In BERT, the remaining 20% are replaced with random tokens or 10% replaced with
    # random tokens and 10% left unchanged.
    # (2.) The masking has two sample phases, one with `self.masked_rate` and one with
    # hardcoded `0.8 * self.masked_rate`. This means that the actual masking rate
    # becomes `0.8 * self.masked_rate` which seems confusing and possibly not intended.
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Get a single masked sequence sample.

        Parameters
        ----------
        index : int
            Index of the sequence to retrieve.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            A tuple of tensors containing input sequence with masked tokens, target
            sequence with non-masked positions set to 0, original input sequence, and
            original target sequence, respectively.
        """
        x = torch.tensor(self.x[index], dtype=torch.int64)
        y = torch.tensor(self.y[index], dtype=torch.int64)

        x_masked = x.clone().detach()
        y_masked = x.clone().detach()

        # non-padding positions (0 is padding)
        seq_len = torch.sum(x_masked > 0)
        # positions to mask
        valid_positions = self.box[x_masked > 0].tolist()
        n_to_mask = int(seq_len * self.masked_rate)

        # randomly sample positions to mask
        mask_positions = random.sample(valid_positions, n_to_mask)
        no_mask_positions = [
            pos for pos in valid_positions if pos not in mask_positions
        ]

        # apply masking
        actual_mask_positions = random.sample(
            mask_positions, int(len(mask_positions) * 0.8)
        )
        x_masked[actual_mask_positions] = self.mask_idx

        # for RNA, also mask adjacent nucleotides for base pairing
        if self.is_rna:
            x_masked = self._mask_rna(x_masked, actual_mask_positions)

        # zero out non-masked positions in target
        y_masked[no_mask_positions] = 0

        return x_masked, y_masked, x, y
