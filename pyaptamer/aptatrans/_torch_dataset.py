"""Private torch Dataset wrapper for AptaTrans training.

This class is internal to the ``aptatrans`` module. It receives
already-encoded numeric arrays (the output of ``rna2vec`` for aptamers
and ``encode_rna`` / similar for proteins) and adapts them for
torch DataLoader consumption via per-sample tensor casting.

Not part of the public API. Do not import from outside ``pyaptamer.aptatrans``.
"""

__author__ = ["siddharth7113"]
__all__ = ["_AptaTransTorchDataset"]

import numpy as np
import torch
from torch.utils.data import Dataset


class _AptaTransTorchDataset(Dataset):
    """Per-sample tensor cast for encoded aptamer-protein pairs.

    Parameters
    ----------
    x_apta : np.ndarray
        Encoded aptamer sequences, shape (n, apta_len).
    x_prot : np.ndarray
        Encoded protein sequences, shape (n, prot_len).
    y : np.ndarray, optional
        Integer labels (e.g., 0/1). If None, ``__getitem__`` returns None
        for the label position.

    Notes
    -----
    This class expects **already-encoded** numeric arrays and **integer labels**.
    Preprocessing steps like data augmentation (e.g., ``augment_reverse``) and
    label encoding (e.g., ``"positive"`` → ``1``) are the caller's responsibility
    and should be applied before constructing this dataset.
    """

    def __init__(self, x_apta, x_prot, y=None):
        self.x_apta = np.asarray(x_apta)
        self.x_prot = np.asarray(x_prot)
        self.y = np.asarray(y) if y is not None else None

    def __len__(self):
        return len(self.x_apta)

    def __getitem__(self, idx):
        x_a = torch.tensor(self.x_apta[idx])
        x_p = torch.tensor(self.x_prot[idx])
        if self.y is None:
            return x_a, x_p
        return x_a, x_p, torch.tensor(self.y[idx])
