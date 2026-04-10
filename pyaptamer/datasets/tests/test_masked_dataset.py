"""Tests for MaskedDataset."""

import pytest

from pyaptamer.datasets.dataclasses import MaskedDataset


@pytest.fixture
def basic_dataset():
    """Create a basic MaskedDataset for testing."""
    x = [[1, 2, 3, 4, 0], [2, 1, 4, 3, 0]]
    y = [[1, 2, 3, 4, 0], [2, 1, 4, 3, 0]]
    return MaskedDataset(x, y, max_len=5, mask_idx=5, masked_rate=0.2)


def test_masked_dataset_length(basic_dataset):
    """Check __len__ returns correct count."""
    assert len(basic_dataset) == 2


def test_masked_dataset_getitem_returns_four_tensors(basic_dataset):
    """Check __getitem__ returns four tensors."""
    result = basic_dataset[0]
    assert len(result) == 4


def test_masked_dataset_mismatched_lengths():
    """Mismatched x and y lengths should raise."""
    x = [[1, 2, 3]]
    y = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(ValueError, match="same length"):
        MaskedDataset(x, y, max_len=3, mask_idx=5)


def test_y_masked_initialized_from_y_not_x():
    """Check y_masked is derived from y, not from x."""
    x = [[1, 2, 3, 4, 0]]
    y = [[5, 6, 7, 8, 0]]

    dataset = MaskedDataset(x, y, max_len=5, mask_idx=9, masked_rate=0.5, is_rna=False)

    _, y_masked, _, _ = dataset[0]

    nonzero_mask = y_masked != 0
    if nonzero_mask.any():
        y_vals = set(y[0])
        for val in y_masked[nonzero_mask].tolist():
            assert val in y_vals
