"""Private torch Dataset wrapper for AptaTrans training.

This class is internal to the ``aptatrans`` module. It receives
already-encoded numeric arrays (the output of ``rna2vec`` for aptamers
and ``encode_rna`` / similar for proteins) and adapts them for
torch DataLoader consumption with optional per-sample augmentation.

Not part of the public API. Do not import from outside ``pyaptamer.aptatrans``.
"""

__all__ = ["_AptaTransTorchDataset"]

import numpy as np
import torch
from torch.utils.data import Dataset


class _AptaTransTorchDataset(Dataset):
    """Per-sample tensor cast + optional train-time augmentation.

    Parameters
    ----------
    x_apta : np.ndarray
        Encoded aptamer sequences, shape (n, apta_len).
    x_prot : np.ndarray
        Encoded protein sequences, shape (n, prot_len).
    y : np.ndarray, optional
        Labels (e.g., 0/1). If None, ``__getitem__`` returns None for the
        label position.
    augment : bool, default False
        If True, applies per-sample reverse-complement augmentation to the
        aptamer at ``__getitem__`` time. Intended for training loops; set
        False for validation/test/inference.
    """

    def __init__(self, x_apta, x_prot, y=None, augment=False):
        self.x_apta = np.asarray(x_apta)
        self.x_prot = np.asarray(x_prot)
        self.y = np.asarray(y) if y is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.x_apta)

    def __getitem__(self, i):
        x_a = self.x_apta[i]
        x_p = self.x_prot[i]
        if self.augment:
            x_a = self._augment_one(x_a)
        x_a_t = torch.tensor(x_a)
        x_p_t = torch.tensor(x_p)
        if self.y is None:
            return x_a_t, x_p_t, None
        y_t = torch.tensor(self.y[i])
        return x_a_t, x_p_t, y_t

    @staticmethod
    def _augment_one(x_a):
        """Per-sample reverse-complement augmentation hook.

        Default implementation: reverses the encoded sequence (a stand-in
        for proper reverse-complement on encoded form). The full
        reverse-complement semantics live in
        ``pyaptamer.utils._augment.augment_reverse`` and operate on string
        arrays; integrating that for the encoded form is the responsibility
        of the stacked AptaTrans fit/predict PR.
        """
        return x_a[::-1].copy()
