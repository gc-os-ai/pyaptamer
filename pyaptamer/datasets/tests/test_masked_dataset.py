"""Tests for MaskedDataset."""

import random

import pytest
import torch

from pyaptamer.datasets.dataclasses import MaskedDataset


class TestMaskedDataset:
    """Tests for MaskedDataset."""

    def test_len(self):
        """Check that __len__ returns the correct number of sequences."""
        x = [[1, 2, 3, 0], [4, 5, 6, 0]]
        y = [[1, 2, 3, 0], [4, 5, 6, 0]]
        ds = MaskedDataset(x, y, max_len=4, mask_idx=99)
        assert len(ds) == 2

    def test_mismatched_lengths_raises(self):
        """Check that mismatched x and y lengths raise ValueError."""
        x = [[1, 2, 3]]
        y = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match="same length"):
            MaskedDataset(x, y, max_len=3, mask_idx=99)

    def test_getitem_returns_four_tensors(self):
        """Check that __getitem__ returns a tuple of four tensors."""
        x = [[1, 2, 3, 4, 0]]
        y = [[1, 2, 3, 4, 0]]
        ds = MaskedDataset(x, y, max_len=5, mask_idx=99, masked_rate=0.5)
        result = ds[0]
        assert len(result) == 4
        assert all(isinstance(t, torch.Tensor) for t in result)

    def test_getitem_y_masked_derived_from_y(self):
        """Check that y_masked is derived from y, not x.

        This is the regression test for the bug where y_masked was
        initialized from x.clone() instead of y.clone().
        """
        # Use clearly different x and y values so the source is detectable
        x = [[1, 2, 3, 4, 5, 0, 0, 0]]
        y = [[10, 20, 30, 40, 50, 0, 0, 0]]
        ds = MaskedDataset(x, y, max_len=8, mask_idx=99, masked_rate=0.5)

        torch.manual_seed(42)
        random.seed(42)

        x_masked, y_masked, x_orig, y_orig = ds[0]

        # Non-zero values in y_masked must come from y (values >= 10),
        # not from x (values 1-5)
        nonzero_y_masked = y_masked[y_masked > 0]
        if len(nonzero_y_masked) > 0:
            # All non-zero values should be from y's value range
            assert all(v >= 10 for v in nonzero_y_masked), (
                f"y_masked contains values from x instead of y: {y_masked}"
            )
