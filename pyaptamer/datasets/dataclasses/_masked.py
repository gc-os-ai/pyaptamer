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

    This dataset implements BERT-style random masking for sequence data [1]_. For each
    sample, ``masked_rate`` of non-padding tokens are selected. Of those selected
    tokens:

    - **80 %** are replaced with the ``mask_idx`` token,
    - **10 %** are replaced with a uniformly random token drawn from
      ``[1, vocab_size]`` (only when ``vocab_size`` is provided),
    - **10 %** are left unchanged.

    The training target (``y_masked``) retains the original token values at all
    selected positions and is zero elsewhere, so the loss is computed only over the
    selected tokens. For RNA sequences, positions adjacent to masked tokens are also
    masked to account for base-pairing interactions.

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
        Proportion of non-padding tokens to select for masking
        (should be between 0.0 and 1.0).
    is_rna : bool, optional, default=False
        Whether the sequences are RNA (True) or DNA (False). For RNA sequences,
        adjacent nucleotides are also masked to account for base pairing.
        Default is False.
    vocab_size : int or None, optional, default=None
        Size of the token vocabulary (excluding the padding token 0). When provided,
        the 10 % random-replacement step draws tokens uniformly from
        ``[1, vocab_size]``. When ``None``, the random-replacement step is skipped
        and those positions are left unchanged instead.

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

    References
    ----------
    .. [1] Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional
       Transformers for Language Understanding." NAACL 2019.

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
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()

        if len(x) != len(y):
            raise ValueError(
                f"Input and target arrays must have the same length. "
                f"Got x: {len(x)}, y: {len(y)}"
            )

        self.x, self.y = np.array(x), np.array(y)
        self.max_len = max_len
        self.mask_idx = mask_idx
        self.masked_rate = masked_rate
        self.is_rna = is_rna
        self.vocab_size = vocab_size

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

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Get a single masked sequence sample.

        Applies BERT-style masking: ``masked_rate`` of non-padding tokens are
        selected; 80 % are replaced with ``mask_idx``, 10 % with a random token
        (when ``vocab_size`` is set), and 10 % are left unchanged. The returned
        target tensor retains original values only at selected positions so the
        pre-training loss is computed exclusively over those tokens.

        Parameters
        ----------
        index : int
            Index of the sequence to retrieve.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            A tuple of ``(x_masked, y_masked, x, y)`` where:

            - ``x_masked``: input sequence with selected tokens replaced/masked.
            - ``y_masked``: original sequence with non-selected positions zeroed.
            - ``x``: original (unmodified) input sequence.
            - ``y``: original target sequence.
        """
        x = torch.tensor(self.x[index], dtype=torch.int64)
        y = torch.tensor(self.y[index], dtype=torch.int64)

        x_masked = x.clone().detach()
        y_masked = x.clone().detach()

        # non-padding positions (0 is padding)
        seq_len = int(torch.sum(x_masked > 0).item())
        n_to_mask = int(seq_len * self.masked_rate)

        if n_to_mask == 0:
            y_masked[:] = 0
            return x_masked, y_masked, x, y

        valid_positions = self.box[(x_masked > 0).numpy()].tolist()

        # select positions to process (BERT: `masked_rate` of all non-padding tokens)
        selected_positions = random.sample(valid_positions, n_to_mask)

        # BERT 80/10/10 split over the selected positions
        n_mask_token = int(n_to_mask * 0.8)
        n_random_token = int(n_to_mask * 0.1)
        # the remaining positions are left unchanged in x_masked

        shuffled = selected_positions.copy()
        random.shuffle(shuffled)
        mask_token_positions = shuffled[:n_mask_token]
        random_token_positions = shuffled[n_mask_token : n_mask_token + n_random_token]

        # 80 % → replace with [MASK]
        x_masked[mask_token_positions] = self.mask_idx

        # 10 % → replace with a uniformly random token
        if self.vocab_size is not None:
            for pos in random_token_positions:
                x_masked[pos] = random.randint(1, self.vocab_size)

        # for RNA, also mask adjacent nucleotides for base pairing
        if self.is_rna:
            x_masked = self._mask_rna(x_masked, mask_token_positions)

        # zero out non-selected positions in target so loss is only over selected tokens
        selected_set = set(selected_positions)
        no_mask_positions = [pos for pos in valid_positions if pos not in selected_set]
        y_masked[no_mask_positions] = 0

        return x_masked, y_masked, x, y
