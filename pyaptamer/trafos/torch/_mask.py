"""Tensor-to-tensor transformations."""

import random

from torch import Tensor

from pyaptamer.trafos.torch._base import BaseTorchTransform


class RandomMask(BaseTorchTransform):
    """Randomly mask positions in a sequence tensor.
    
    Parameters
    ----------
    mask_idx : int
        Token ID to use for masked positions.
    mask_rate : float, default=0.15
        Proportion of valid (non-padded) positions to randomly mask.
    padding_idx : int, default=0
        Token ID representing padding (positions to skip).
    
    Usage
    -----
    >>> import torch
    >>> from pyaptamer.trafos.torch import GreedyEncode, RandomMask
    >>> vocab = {"A": 1, "T": 2, "C": 3, "G": 4}
    >>> encoder = GreedyEncode(vocab=vocab, max_len=16)
    >>> encoded = encoder("ATCG")
    >>> masker = RandomMask(mask_idx=999, mask_rate=0.15)
    >>> masked = masker(encoded)
    >>> (masked != encoded).sum()  # Some positions were masked
    tensor(1)
    """

    def __init__(self, mask_idx: int, mask_rate: float = 0.15, padding_idx: int = 0):
        self.mask_idx = mask_idx
        self.mask_rate = mask_rate
        self.padding_idx = padding_idx

    def __call__(self, x: Tensor) -> Tensor:
        x_masked = x.clone()
        valid_pos = (x != self.padding_idx).nonzero(as_tuple=True)[0].tolist()
        n_mask = int(len(valid_pos) * self.mask_rate)

        if n_mask > 0:
            mask_pos = random.sample(valid_pos, n_mask)
            x_masked[mask_pos] = self.mask_idx

        return x_masked

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mask_idx={self.mask_idx})"
