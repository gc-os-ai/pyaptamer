"""String-to-tensor transformations."""

import torch
from torch import Tensor

from pyaptamer.trafos.torch._base import BaseTorchTransform


class GreedyEncode(BaseTorchTransform):
    """Greedy tokenization of sequence to tensor.

    Matches longest possible token at each position. Unknown chars map to 0.
    
    Parameters
    ----------
    vocab : dict[str, int]
        Token to integer ID mapping.
    max_len : int
        Maximum padded sequence length.
    token_max_len : int, optional
        Maximum token length to match. Defaults to longest vocab key.
    
    Usage
    -----
    >>> vocab = {"A": 1, "T": 2, "C": 3, "G": 4, "AT": 5}
    >>> encoder = GreedyEncode(vocab=vocab, max_len=16)
    >>> encoded = encoder("ATGC")
    >>> encoded.shape
    torch.Size([16])
    >>> encoded.dtype
    torch.int64
    """

    def __init__(self, vocab: dict[str, int], max_len: int, token_max_len: int = None):
        self.vocab = vocab
        self.max_len = max_len
        self.token_max_len = token_max_len or max(len(k) for k in vocab)

    def __call__(self, x: str) -> Tensor:
        tokens = []
        i = 0
        while i < len(x):
            matched = False
            for j in range(self.token_max_len, 0, -1):
                if i + j <= len(x):
                    substr = x[i : i + j]
                    if substr in self.vocab:
                        tokens.append(self.vocab[substr])
                        i += j
                        matched = True
                        break
            if not matched:
                tokens.append(0)
                i += 1

        if len(tokens) < self.max_len:
            tokens.extend([0] * (self.max_len - len(tokens)))
        else:
            tokens = tokens[: self.max_len]

        return torch.tensor(tokens, dtype=torch.long)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_len={self.max_len})"
