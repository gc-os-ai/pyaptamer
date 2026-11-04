"""Tests for MaskedDataset."""

__author__ = ["anshuman83-40"]

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pyaptamer.datasets.dataclasses import MaskedDataset

MASK_IDX = 9


@pytest.fixture
def padded_seqs():
    """Two sequences of length 10, the second with trailing padding."""
    x = [
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
        [4, 3, 2, 1, 4, 3, 0, 0, 0, 0],
    ]
    return x, [list(s) for s in x]


def test_mismatched_lengths_raise():
    """x and y of different lengths raise a ValueError."""
    with pytest.raises(ValueError, match="same length"):
        MaskedDataset([[1, 2]], [[1, 2], [2, 1]], max_len=2, mask_idx=MASK_IDX)


@pytest.mark.parametrize("as_array", [False, True])
def test_init_attributes(padded_seqs, as_array):
    """Constructor stores inputs as arrays and sets box and len."""
    x, y = padded_seqs
    if as_array:
        x, y = np.array(x), np.array(y)

    ds = MaskedDataset(x, y, max_len=10, mask_idx=MASK_IDX, masked_rate=0.3)

    assert isinstance(ds.x, np.ndarray)
    assert isinstance(ds.y, np.ndarray)
    np.testing.assert_array_equal(ds.box, np.arange(10))
    assert ds.len == 2
    assert len(ds) == 2
    assert ds.max_len == 10
    assert ds.mask_idx == MASK_IDX
    assert ds.masked_rate == 0.3
    assert ds.is_rna is False


@pytest.mark.parametrize("is_rna", [False, True])
def test_getitem_output_contract(padded_seqs, is_rna):
    """__getitem__ returns four int64 tensors of length max_len."""
    x, y = padded_seqs
    ds = MaskedDataset(x, y, max_len=10, mask_idx=MASK_IDX, is_rna=is_rna)

    for i in range(len(ds)):
        out = ds[i]
        assert len(out) == 4
        for t in out:
            assert isinstance(t, torch.Tensor)
            assert t.dtype == torch.int64
            assert t.shape == (10,)

        _, _, x_orig, y_orig = out
        assert torch.equal(x_orig, torch.tensor(x[i]))
        assert torch.equal(y_orig, torch.tensor(y[i]))


def test_getitem_does_not_mutate_stored_data(padded_seqs):
    """Masking works on copies and leaves the stored arrays untouched."""
    x, y = padded_seqs
    ds = MaskedDataset(x, y, max_len=10, mask_idx=MASK_IDX, masked_rate=1.0)

    ds[0]
    ds[1]

    np.testing.assert_array_equal(ds.x, np.array(x))
    np.testing.assert_array_equal(ds.y, np.array(y))


def test_zero_masked_rate_masks_nothing(padded_seqs):
    """With masked_rate=0, the input is unchanged and the target is all zeros."""
    x, y = padded_seqs
    ds = MaskedDataset(x, y, max_len=10, mask_idx=MASK_IDX, masked_rate=0.0)

    x_masked, y_masked, x_orig, _ = ds[0]

    assert torch.equal(x_masked, x_orig)
    assert torch.count_nonzero(y_masked) == 0


def test_dna_masking_counts_and_positions(padded_seqs):
    """DNA masking hits 80% of the sampled positions and never touches padding."""
    x, y = padded_seqs
    ds = MaskedDataset(x, y, max_len=10, mask_idx=MASK_IDX, masked_rate=0.5)

    random.seed(0)
    for i, n_valid in [(0, 10), (1, 6)]:
        x_masked, y_masked, x_orig, _ = ds[i]

        n_to_mask = int(n_valid * 0.5)
        n_masked = int(n_to_mask * 0.8)
        is_masked = x_masked == MASK_IDX

        assert int(is_masked.sum()) == n_masked
        # unmasked positions keep their original value
        assert torch.equal(x_masked[~is_masked], x_orig[~is_masked])
        # padding stays padding
        assert torch.all(x_masked[x_orig == 0] == 0)

        # target keeps the original tokens only at the sampled positions
        kept = y_masked != 0
        assert int(kept.sum()) == n_to_mask
        assert torch.equal(y_masked[kept], x_orig[kept])
        # every actually-masked position is part of the target
        assert torch.all(kept[is_masked])


def test_dna_full_rate_masks_eighty_percent():
    """With masked_rate=1.0 on a DNA sequence, int(0.8 * len) positions are masked."""
    x = [[1, 2, 3, 4, 1]]
    ds = MaskedDataset(x, x, max_len=5, mask_idx=MASK_IDX, masked_rate=1.0)

    x_masked, y_masked, x_orig, _ = ds[0]

    assert int((x_masked == MASK_IDX).sum()) == 4
    # every position was sampled, so the target is the full input
    assert torch.equal(y_masked, x_orig)


def test_rna_full_rate_masks_neighbours_too():
    """In RNA mode, neighbours of masked positions are masked as well."""
    x = [[1, 2, 3, 4, 1]]
    ds = MaskedDataset(x, x, max_len=5, mask_idx=MASK_IDX, masked_rate=1.0, is_rna=True)

    for seed in range(10):
        random.seed(seed)
        x_masked, _, _, _ = ds[0]
        # 4 of 5 positions are masked directly, the remaining one is a neighbour
        assert torch.all(x_masked == MASK_IDX)


def test_rna_masks_superset_of_dna():
    """For the same random state, RNA masking covers every DNA-masked position."""
    x = [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    kwargs = {"max_len": 12, "mask_idx": MASK_IDX, "masked_rate": 0.5}
    dna = MaskedDataset(x, x, **kwargs)
    rna = MaskedDataset(x, x, is_rna=True, **kwargs)

    for seed in range(10):
        random.seed(seed)
        dna_mask = dna[0][0] == MASK_IDX
        random.seed(seed)
        rna_mask = rna[0][0] == MASK_IDX

        assert torch.all(rna_mask[dna_mask])
        assert rna_mask.sum() > dna_mask.sum()

        # every RNA-masked position is a DNA-masked position or its neighbour
        dna_pos = set(torch.nonzero(dna_mask).flatten().tolist())
        allowed = dna_pos | {p + 1 for p in dna_pos} | {p - 1 for p in dna_pos}
        assert set(torch.nonzero(rna_mask).flatten().tolist()) <= allowed


@pytest.mark.parametrize(
    "positions, expected",
    [
        ([2], [1, 2, 3]),  # interior: both neighbours
        ([0], [0, 1]),  # first position: right neighbour only
        ([4], [3, 4]),  # last position: left neighbour only
        ([0, 4], [0, 1, 3, 4]),
        ([], []),  # nothing to mask
    ],
)
def test_mask_rna_neighbours_and_bounds(positions, expected):
    """_mask_rna masks the neighbours of each position and respects the bounds."""
    ds = MaskedDataset(
        [[1, 2, 3, 4, 1]], [[1, 2, 3, 4, 1]], max_len=5, mask_idx=MASK_IDX
    )
    x_masked = torch.tensor([1, 2, 3, 4, 1])
    x_masked[positions] = MASK_IDX

    out = ds._mask_rna(x_masked, positions)

    assert sorted(torch.nonzero(out == MASK_IDX).flatten().tolist()) == expected


def test_seed_reproducibility(padded_seqs):
    """Masking depends only on Python's random state."""
    x, y = padded_seqs
    ds = MaskedDataset(x, y, max_len=10, mask_idx=MASK_IDX, masked_rate=0.5)

    random.seed(42)
    first = ds[0]
    random.seed(42)
    second = ds[0]

    for a, b in zip(first, second, strict=False):
        assert torch.equal(a, b)


def test_dataloader_batching(padded_seqs):
    """The dataset works with a torch DataLoader."""
    x, y = padded_seqs
    ds = MaskedDataset(x, y, max_len=10, mask_idx=MASK_IDX)

    batch = next(iter(DataLoader(ds, batch_size=2)))

    assert len(batch) == 4
    for t in batch:
        assert t.shape == (2, 10)
