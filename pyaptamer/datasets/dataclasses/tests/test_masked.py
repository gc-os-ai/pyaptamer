"""Tests for MaskedDataset behavior not covered by its docstring example."""

from pyaptamer.datasets.dataclasses import MaskedDataset


def test_masked_dataset_getitem_returns_4_tuple():
    """__getitem__ returns (x_masked, y_masked, x, y) tensors.

    The docstring example only verifies len(); this confirms the per-sample
    output shape used by the DataLoader.
    """
    sequences = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    targets = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    ds = MaskedDataset(
        sequences, targets, max_len=5, mask_idx=5, masked_rate=0.2, is_rna=True
    )
    sample = ds[0]
    assert len(sample) == 4
