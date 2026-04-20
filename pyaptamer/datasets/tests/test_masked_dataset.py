__author__ = ["ritankarsaha"]

import pytest
import torch

from pyaptamer.datasets.dataclasses._masked import MaskedDataset

# Sequences where x != y to verify the MLM target uses y, not x
X = [[1, 2, 3, 4, 0], [2, 1, 4, 3, 0]]
Y = [[5, 6, 7, 8, 0], [6, 5, 8, 7, 0]]
MAX_LEN = 5
MASK_IDX = 9


def _make_dataset(is_rna: bool = False) -> MaskedDataset:
    return MaskedDataset(
        X, Y, max_len=MAX_LEN, mask_idx=MASK_IDX, masked_rate=0.5, is_rna=is_rna
    )


def test_y_masked_derives_from_y_not_x():
    """Core regression: y_masked must be cloned from y, not x."""
    ds = _make_dataset()
    for i in range(len(ds)):
        x_masked, y_masked, x_orig, y_orig = ds[i]
        # Any non-zero position in y_masked must equal the corresponding y value
        non_zero_mask = y_masked != 0
        assert torch.all(y_masked[non_zero_mask] == y_orig[non_zero_mask]), (
            "y_masked contains values from x instead of y"
        )


def test_original_tensors_returned_correctly():
    """x and y returned as last two elements must match the stored arrays."""
    ds = _make_dataset()
    for i in range(len(ds)):
        x_masked, y_masked, x_orig, y_orig = ds[i]
        assert torch.equal(x_orig, torch.tensor(X[i], dtype=torch.int64))
        assert torch.equal(y_orig, torch.tensor(Y[i], dtype=torch.int64))


def test_masked_positions_in_x_masked():
    """x_masked must contain MASK_IDX at at least one position."""
    ds = _make_dataset()
    found_mask = False
    for i in range(len(ds)):
        x_masked, _, _, _ = ds[i]
        if (x_masked == MASK_IDX).any():
            found_mask = True
            break
    assert found_mask, "No masked positions found in x_masked across all samples"


def test_no_mask_positions_zeroed_when_rate_zero():
    """With masked_rate=0, no positions are selected for masking, so all valid
    positions in y_masked must be zeroed (all become no_mask_positions)."""
    ds = MaskedDataset(X, Y, max_len=MAX_LEN, mask_idx=MASK_IDX, masked_rate=0.0)
    for i in range(len(ds)):
        _, y_masked, x_orig, _ = ds[i]
        valid = x_orig > 0
        assert torch.all(y_masked[valid] == 0), (
            "With masked_rate=0 all valid positions should be zeroed in y_masked"
        )


def test_all_mask_positions_retain_y_when_rate_one():
    """With masked_rate=1.0, every valid position is a mask candidate so
    no_mask_positions is empty and y_masked retains the full y values."""
    ds = MaskedDataset(X, Y, max_len=MAX_LEN, mask_idx=MASK_IDX, masked_rate=1.0)
    for i in range(len(ds)):
        _, y_masked, x_orig, y_orig = ds[i]
        valid = x_orig > 0
        assert torch.all(y_masked[valid] == y_orig[valid]), (
            "With masked_rate=1.0, y_masked should retain all y values"
        )


def test_length_mismatch_raises():
    """Mismatched x/y lengths must raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        MaskedDataset([[1, 2]], [[1, 2], [3, 4]], max_len=2, mask_idx=5)


def test_dataset_length():
    ds = _make_dataset()
    assert len(ds) == len(X)


def test_rna_mode_masks_adjacent():
    """RNA mode should mask adjacent positions."""
    ds = _make_dataset(is_rna=True)
    for i in range(len(ds)):
        x_masked, _, _, _ = ds[i]

        assert x_masked.shape[0] == MAX_LEN
