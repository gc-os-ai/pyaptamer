"""Tests for _AptaTransTorchDataset (private model-owned torch Dataset)."""

__author__ = ["siddharth7113"]

import numpy as np
import torch

from pyaptamer.aptatrans._torch_dataset import _AptaTransTorchDataset


def _toy_encoded():
    """Simulate already-encoded sequences (random ints).

    Mimics the output of rna2vec (aptamers) and encode_rna (proteins):
    integer arrays of shape (n_samples, sequence_length).
    """
    rng = np.random.default_rng(0)
    x_apta = rng.integers(0, 100, size=(4, 10))
    x_prot = rng.integers(0, 50, size=(4, 8))
    y = np.array([1, 0, 1, 0])
    return x_apta, x_prot, y


def test_len_matches_input():
    x_apta, x_prot, y = _toy_encoded()
    ds = _AptaTransTorchDataset(x_apta, x_prot, y)
    assert len(ds) == 4


def test_getitem_returns_tensor_triple():
    x_apta, x_prot, y = _toy_encoded()
    ds = _AptaTransTorchDataset(x_apta, x_prot, y)
    sample = ds[0]
    assert len(sample) == 3
    assert all(isinstance(t, torch.Tensor) for t in sample)


def test_getitem_y_value_correct():
    x_apta, x_prot, y = _toy_encoded()
    ds = _AptaTransTorchDataset(x_apta, x_prot, y)
    _, _, y0 = ds[0]
    assert y0.item() == 1
    _, _, y1 = ds[1]
    assert y1.item() == 0


def test_unlabeled_returns_y_none():
    x_apta, x_prot, _ = _toy_encoded()
    ds = _AptaTransTorchDataset(x_apta, x_prot, y=None)
    sample = ds[0]
    assert len(sample) == 2


def test_dataloader_compatibility():
    """Verify torch DataLoader can iterate the dataset without crashing."""
    from torch.utils.data import DataLoader

    x_apta, x_prot, y = _toy_encoded()
    ds = _AptaTransTorchDataset(x_apta, x_prot, y)
    loader = DataLoader(ds, batch_size=2)
    batches = list(loader)
    assert len(batches) == 2
    x_a, x_p, y_b = batches[0]
    assert x_a.shape == (2, 10)
    assert x_p.shape == (2, 8)
    assert y_b.shape == (2,)


def test_no_augment_param():
    """_AptaTransTorchDataset no longer accepts an augment parameter.
    Augmentation is the caller's responsibility (applied before encoding)."""
    x_apta, x_prot, y = _toy_encoded()
    # Should work without augment kwarg
    ds = _AptaTransTorchDataset(x_apta, x_prot, y)
    assert len(ds) == 4
