"""Test suite for MaskedDataset."""

import random

import numpy as np
import pytest
import torch

from pyaptamer.datasets.dataclasses import MaskedDataset


def test_masked_dataset_length_mismatch():
    """Constructor must reject ``x`` and ``y`` of different lengths."""
    with pytest.raises(ValueError, match="must have the same length"):
        MaskedDataset(
            x=[[1, 2], [3, 4]],
            y=[[1, 2]],
            max_len=2,
            mask_idx=9,
        )


def test_masked_dataset_basic_lengths_and_types():
    """``__len__`` and ``__getitem__`` return the documented shapes/types."""
    sequences = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]
    targets = [[1, 2, 3, 4, 0], [2, 1, 4, 0, 0]]

    ds = MaskedDataset(
        sequences, targets, max_len=5, mask_idx=9, masked_rate=0.2, is_rna=False
    )
    assert len(ds) == 2

    random.seed(0)
    x_masked, y_masked, x, y = ds[0]
    assert x_masked.shape == (5,)
    assert y_masked.shape == (5,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    # Padding positions in input must remain padding (0).
    assert x_masked[-1].item() == 0


def test_mask_rna_does_not_overwrite_padding():
    """RNA adjacent masking must not turn padding tokens into ``mask_idx``.

    Reproduces the bug in which ``_mask_rna`` used ``self.max_len`` as the
    upper bound when picking right-neighbour positions, so the slot just past
    the actual sequence (the first padding token) could be masked.
    """
    sequences = [[1, 2, 3, 0, 0]]
    targets = [[1, 2, 3, 0, 0]]

    ds = MaskedDataset(
        sequences, targets, max_len=5, mask_idx=99, masked_rate=1.0, is_rna=True
    )

    # Sweep enough seeds to make sure the last non-padding position
    # (index 2) is selected for masking at least once. With masked_rate=1.0
    # every non-padding position is sampled, so this is guaranteed on the
    # first iteration; the loop simply guards against any future RNG change.
    for seed in range(20):
        random.seed(seed)
        x_masked, _, x, _ = ds[0]
        assert x[3].item() == 0  # sanity: original was padding
        assert x[4].item() == 0
        # Padding tokens must remain padding after masking.
        assert x_masked[3].item() == 0, (
            f"seed={seed}: padding at index 3 was overwritten with mask_idx"
        )
        assert x_masked[4].item() == 0


def test_mask_rna_still_masks_internal_neighbours():
    """Adjacent masking must still apply for positions inside the sequence."""
    sequences = [[1, 2, 3, 4, 5]]
    targets = [[1, 2, 3, 4, 5]]

    ds = MaskedDataset(
        sequences, targets, max_len=5, mask_idx=99, masked_rate=1.0, is_rna=True
    )

    # When every non-padding position is in mask_positions, the right-
    # neighbour and left-neighbour rules together cover the whole sequence.
    random.seed(0)
    x_masked, _, _, _ = ds[0]
    assert (x_masked == 99).all().item()


def test_mask_rna_unit_helper_respects_seq_len():
    """Direct unit test on the ``_mask_rna`` helper for the bounds fix."""
    ds = MaskedDataset(
        x=[[1, 2, 3, 0, 0]],
        y=[[1, 2, 3, 0, 0]],
        max_len=5,
        mask_idx=99,
        masked_rate=1.0,
        is_rna=True,
    )

    x_masked = torch.tensor([1, 2, 99, 0, 0], dtype=torch.int64)
    out = ds._mask_rna(x_masked.clone(), mask_positions=[2], seq_len=3)
    # left neighbour (1) is masked; right neighbour (3) is padding -> unchanged
    assert out[1].item() == 99
    assert out[3].item() == 0
    assert out[4].item() == 0


def test_y_masked_zeroed_at_no_mask_positions():
    """``y_masked`` must zero out positions that were never selected for masking.

    Positions sampled into ``mask_positions`` keep their original target value
    (whether or not they end up actually replaced with ``mask_idx`` after the
    80% sub-sample), while every other valid position is zeroed in y.
    """
    sequences = np.array([[1, 2, 3, 4, 5]])
    targets = np.array([[1, 2, 3, 4, 5]])
    ds = MaskedDataset(
        sequences, targets, max_len=5, mask_idx=99, masked_rate=0.4, is_rna=False
    )
    random.seed(0)
    _, y_masked, x, _ = ds[0]
    # At least one position must be zeroed (since masked_rate < 1).
    assert (y_masked == 0).any().item()
    # Wherever y_masked is non-zero, it equals x.
    nz = y_masked != 0
    assert torch.equal(y_masked[nz], x[nz])
