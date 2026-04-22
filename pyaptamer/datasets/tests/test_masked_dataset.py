import numpy as np
import pytest

from pyaptamer.datasets.dataclasses import MaskedDataset


@pytest.fixture
def dataset():
    x = [[1, 2, 3, 4, 0], [2, 1, 4, 3, 0]]
    y = [[5, 6, 7, 8, 0], [6, 5, 8, 7, 0]]
    return MaskedDataset(x, y, max_len=5, mask_idx=9, masked_rate=0.2)


def test_getitem_returns_four_tensors(dataset):
    x_masked, y_masked, x, y = dataset[0]
    assert x_masked.shape == (5,)
    assert y_masked.shape == (5,)
    assert x.shape == (5,)
    assert y.shape == (5,)


def test_y_masked_derived_from_y_not_x(dataset):
    """y_masked should contain values from y, not from x."""
    x_masked, y_masked, x_orig, y_orig = dataset[0]
    non_zero = y_masked[y_masked != 0]
    for val in non_zero:
        assert val.item() in y_orig.tolist()


def test_original_tensors_match_input(dataset):
    _, _, x, y = dataset[0]
    np.testing.assert_array_equal(x.numpy(), [1, 2, 3, 4, 0])
    np.testing.assert_array_equal(y.numpy(), [5, 6, 7, 8, 0])


def test_length(dataset):
    assert len(dataset) == 2


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        MaskedDataset([[1, 2]], [[1, 2], [3, 4]], max_len=2, mask_idx=9)


def test_rna_masking():
    x = [[1, 2, 3, 4, 5]]
    y = [[1, 2, 3, 4, 5]]
    ds = MaskedDataset(x, y, max_len=5, mask_idx=9, masked_rate=0.5, is_rna=True)
    x_masked, _, _, _ = ds[0]
    assert (x_masked == 9).any()
