"""Tests for the MaskedDataset dataclass."""

__author__ = ["nennomp"]

import pytest
import torch

from pyaptamer.datasets.dataclasses import MaskedDataset


def test_masked_dataset_init_rejects_mismatched_lengths():
    """Input and target arrays must contain the same number of sequences."""
    with pytest.raises(ValueError, match="same length"):
        MaskedDataset(
            x=[[1, 2, 3, 0]],
            y=[[1, 2, 3, 0], [4, 5, 6, 0]],
            max_len=4,
            mask_idx=9,
        )


@pytest.mark.parametrize("max_len", [0, -1])
def test_masked_dataset_init_rejects_non_positive_max_len(max_len):
    """`max_len` must be positive."""
    with pytest.raises(ValueError, match="`max_len` must be a positive integer"):
        MaskedDataset(
            x=[[1, 2, 3, 0]],
            y=[[1, 2, 3, 0]],
            max_len=max_len,
            mask_idx=9,
        )


@pytest.mark.parametrize("masked_rate", [-0.1, 1.1])
def test_masked_dataset_init_rejects_invalid_masked_rate(masked_rate):
    """`masked_rate` must be in the closed interval [0.0, 1.0]."""
    with pytest.raises(ValueError, match="`masked_rate` must be between 0.0 and 1.0"):
        MaskedDataset(
            x=[[1, 2, 3, 0]],
            y=[[1, 2, 3, 0]],
            max_len=4,
            mask_idx=9,
            masked_rate=masked_rate,
        )


def test_masked_dataset_init_rejects_non_2d_inputs():
    """Encoded inputs must be two-dimensional arrays."""
    with pytest.raises(ValueError, match="must be 2D"):
        MaskedDataset(
            x=[1, 2, 3, 0],
            y=[1, 2, 3, 0],
            max_len=4,
            mask_idx=9,
        )


def test_masked_dataset_init_rejects_sequence_length_mismatch():
    """Encoded sequence width must match `max_len`."""
    with pytest.raises(ValueError, match="length equal to `max_len`"):
        MaskedDataset(
            x=[[1, 2, 3, 0]],
            y=[[1, 2, 3, 0]],
            max_len=5,
            mask_idx=9,
        )


def test_masked_dataset_getitem_masks_inputs_and_uses_target(monkeypatch):
    """Masked targets must be derived from `y`, not copied from `x`."""

    def mock_sample(population, k):
        return list(population)[:k]

    monkeypatch.setattr(
        "pyaptamer.datasets.dataclasses._masked.random.sample", mock_sample
    )

    dataset = MaskedDataset(
        x=[[1, 2, 3, 4, 0]],
        y=[[9, 8, 7, 6, 0]],
        max_len=5,
        mask_idx=99,
        masked_rate=0.5,
    )

    x_masked, y_masked, x, y = dataset[0]

    assert torch.equal(x, torch.tensor([1, 2, 3, 4, 0]))
    assert torch.equal(y, torch.tensor([9, 8, 7, 6, 0]))
    assert torch.equal(x_masked, torch.tensor([99, 2, 3, 4, 0]))
    assert torch.equal(y_masked, torch.tensor([9, 8, 0, 0, 0]))


def test_masked_dataset_getitem_masks_adjacent_rna_positions(monkeypatch):
    """RNA masking should also mask immediate neighbors of masked positions."""

    def mock_sample(population, k):
        return list(population)[:k]

    monkeypatch.setattr(
        "pyaptamer.datasets.dataclasses._masked.random.sample", mock_sample
    )

    dataset = MaskedDataset(
        x=[[1, 2, 3, 4, 0]],
        y=[[1, 2, 3, 4, 0]],
        max_len=5,
        mask_idx=77,
        masked_rate=0.5,
        is_rna=True,
    )

    x_masked, y_masked, _, _ = dataset[0]

    assert torch.equal(x_masked, torch.tensor([77, 77, 3, 4, 0]))
    assert torch.equal(y_masked, torch.tensor([1, 2, 0, 0, 0]))
