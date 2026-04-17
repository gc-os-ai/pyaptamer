"""Tests for MaskedDataset as a real subclass of BaseAptamerDataset."""

import numpy as np

from pyaptamer.datasets.dataclasses import MaskedDataset
from pyaptamer.datasets.dataclasses._base import BaseAptamerDataset


def test_masked_dataset_inherits_base():
    assert issubclass(MaskedDataset, BaseAptamerDataset)


def test_masked_dataset_scitype_tag():
    assert MaskedDataset.get_class_tags()["scitype"] == "MaskedSequences"


def test_masked_dataset_inner_mtype_tag():
    """MaskedSequences scitype declares its own canonical inner mtype."""
    assert MaskedDataset.get_class_tags()["X_inner_mtype"] == ["numpy_arrays_pair"]


def test_masked_dataset_existing_behavior_preserved():
    """Existing __init__/__getitem__/__len__ contract must still work."""
    sequences = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    targets = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    ds = MaskedDataset(
        sequences, targets, max_len=5, mask_idx=5, masked_rate=0.2, is_rna=True
    )
    assert len(ds) == 2
    sample = ds[0]
    assert len(sample) == 4


def test_masked_dataset_load_returns_x_y_pair():
    """load() returns (x, y) — the canonical sequence/target arrays."""
    sequences = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    targets = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    ds = MaskedDataset(
        sequences, targets, max_len=5, mask_idx=5, masked_rate=0.2, is_rna=False
    )
    x, y = ds.load()
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape == (2, 5)
    assert y.shape == (2, 5)
