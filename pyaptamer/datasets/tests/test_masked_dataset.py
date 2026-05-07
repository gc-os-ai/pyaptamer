__author__ = ["onkar717"]

import random

import numpy as np
import pytest
import torch

from pyaptamer.datasets.dataclasses import MaskedDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEQ_LEN = 20
VOCAB_SIZE = 10
MASK_IDX = VOCAB_SIZE + 1  # distinct from all valid tokens


def _make_dataset(n=50, masked_rate=0.15, is_rna=False, vocab_size=None):
    rng = np.random.default_rng(42)
    seqs = rng.integers(1, VOCAB_SIZE + 1, size=(n, SEQ_LEN)).tolist()
    return MaskedDataset(
        x=seqs,
        y=seqs,
        max_len=SEQ_LEN,
        mask_idx=MASK_IDX,
        masked_rate=masked_rate,
        is_rna=is_rna,
        vocab_size=vocab_size,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_len():
    ds = _make_dataset(n=10)
    assert len(ds) == 10


def test_mismatched_xy_raises():
    with pytest.raises(ValueError, match="same length"):
        MaskedDataset(x=[[1, 2]], y=[[1, 2], [3, 4]], max_len=2, mask_idx=99)


def test_getitem_returns_four_tensors():
    ds = _make_dataset()
    sample = ds[0]
    assert len(sample) == 4
    for t in sample:
        assert isinstance(t, torch.Tensor)


def test_original_tensors_unchanged():
    """x and y (positions 2 and 3) must be identical to the stored sequences."""
    ds = _make_dataset()
    x_masked, y_masked, x, y = ds[0]
    expected = torch.tensor(ds.x[0], dtype=torch.int64)
    assert torch.equal(x, expected)
    assert torch.equal(y, expected)


# ---------------------------------------------------------------------------
# Bug fix (1): effective masking rate must equal masked_rate, not 0.8 * masked_rate
# ---------------------------------------------------------------------------


def test_masking_rate_is_not_double_sampled():
    """
    Previously the implementation sampled masked_rate positions then took 80 % of
    those, making the true mask rate 0.8 * masked_rate. After the fix, the fraction
    of positions replaced with mask_idx should be approximately 0.8 * masked_rate
    (80 % of the selected masked_rate positions), not 0.8 * 0.8 * masked_rate.
    """
    masked_rate = 0.50  # high rate to get stable statistics
    ds = _make_dataset(n=200, masked_rate=masked_rate, vocab_size=VOCAB_SIZE)

    mask_fractions = []
    for i in range(len(ds)):
        x_masked, _, x, _ = ds[i]
        n_valid = int((x > 0).sum().item())
        n_mask_token = int((x_masked == MASK_IDX).sum().item())
        mask_fractions.append(n_mask_token / n_valid)

    mean_frac = np.mean(mask_fractions)
    # Expected: ~0.80 * masked_rate (80 % of selected are mask tokens)
    expected = 0.8 * masked_rate
    # Allow ±5 pp tolerance
    assert abs(mean_frac - expected) < 0.05, (
        f"Mean mask-token fraction {mean_frac:.3f} far from expected {expected:.3f}. "
        "Double-sampling bug may still be present."
    )


# ---------------------------------------------------------------------------
# Bug fix (2): BERT 80/10/10 split
# ---------------------------------------------------------------------------


def test_bert_80_percent_replaced_with_mask_idx():
    """Roughly 80 % of selected positions should carry mask_idx."""
    masked_rate = 0.50
    ds = _make_dataset(n=200, masked_rate=masked_rate, vocab_size=VOCAB_SIZE)

    mask_ratios = []
    for i in range(len(ds)):
        x_masked, y_masked, x, _ = ds[i]
        # selected positions are those where y_masked != 0
        selected = (y_masked != 0).nonzero(as_tuple=True)[0]
        if len(selected) == 0:
            continue
        n_masked = int((x_masked[selected] == MASK_IDX).sum().item())
        mask_ratios.append(n_masked / len(selected))

    assert abs(np.mean(mask_ratios) - 0.8) < 0.05


def test_bert_10_percent_random_token():
    """Roughly 10 % of selected positions should carry a random (non-mask) token."""
    masked_rate = 0.50
    ds = _make_dataset(n=200, masked_rate=masked_rate, vocab_size=VOCAB_SIZE)

    random_ratios = []
    for i in range(len(ds)):
        x_masked, y_masked, x, _ = ds[i]
        selected = (y_masked != 0).nonzero(as_tuple=True)[0]
        if len(selected) == 0:
            continue
        # random positions: token changed but not to mask_idx
        changed = (x_masked[selected] != x[selected]) & (x_masked[selected] != MASK_IDX)
        random_ratios.append(int(changed.sum().item()) / len(selected))

    assert abs(np.mean(random_ratios) - 0.1) < 0.05


def test_bert_10_percent_unchanged():
    """Roughly 10 % of selected positions should be left unchanged."""
    masked_rate = 0.50
    ds = _make_dataset(n=200, masked_rate=masked_rate, vocab_size=VOCAB_SIZE)

    unchanged_ratios = []
    for i in range(len(ds)):
        x_masked, y_masked, x, _ = ds[i]
        selected = (y_masked != 0).nonzero(as_tuple=True)[0]
        if len(selected) == 0:
            continue
        unchanged = x_masked[selected] == x[selected]
        unchanged_ratios.append(int(unchanged.sum().item()) / len(selected))

    assert abs(np.mean(unchanged_ratios) - 0.1) < 0.05


def test_no_random_replacement_without_vocab_size():
    """Without vocab_size, the 10 % random-replacement step is skipped."""
    masked_rate = 0.50
    ds = _make_dataset(n=100, masked_rate=masked_rate, vocab_size=None)

    for i in range(len(ds)):
        x_masked, y_masked, x, _ = ds[i]
        selected = (y_masked != 0).nonzero(as_tuple=True)[0]
        if len(selected) == 0:
            continue
        # without vocab_size, no position should be changed to something other than mask_idx
        changed_to_non_mask = (x_masked[selected] != x[selected]) & (
            x_masked[selected] != MASK_IDX
        )
        assert not changed_to_non_mask.any(), (
            "Random token replacement should not occur when vocab_size is None."
        )


# ---------------------------------------------------------------------------
# Target tensor (y_masked) properties
# ---------------------------------------------------------------------------


def test_y_masked_zero_at_non_selected_positions():
    """Non-selected valid positions in y_masked must be 0."""
    ds = _make_dataset(n=20, masked_rate=0.15, vocab_size=VOCAB_SIZE)
    for i in range(len(ds)):
        x_masked, y_masked, x, _ = ds[i]
        selected_mask = y_masked != 0
        non_selected_valid = (x > 0) & ~selected_mask
        assert (y_masked[non_selected_valid] == 0).all()


def test_y_masked_original_values_at_selected_positions():
    """Selected positions in y_masked must hold the original token value."""
    ds = _make_dataset(n=20, masked_rate=0.50, vocab_size=VOCAB_SIZE)
    for i in range(len(ds)):
        x_masked, y_masked, x, _ = ds[i]
        selected = (y_masked != 0).nonzero(as_tuple=True)[0]
        assert torch.equal(y_masked[selected], x[selected])


# ---------------------------------------------------------------------------
# RNA adjacent masking
# ---------------------------------------------------------------------------


def test_rna_adjacent_masking_applied():
    """For is_rna=True, neighbours of masked positions also receive mask_idx."""
    rng = np.random.default_rng(0)
    seq = rng.integers(1, VOCAB_SIZE + 1, size=(1, SEQ_LEN)).tolist()
    ds = MaskedDataset(
        x=seq, y=seq, max_len=SEQ_LEN, mask_idx=MASK_IDX,
        masked_rate=0.50, is_rna=True, vocab_size=VOCAB_SIZE,
    )

    found_adjacent = False
    for _ in range(20):
        x_masked, y_masked, x, _ = ds[0]
        mask_positions = (x_masked == MASK_IDX).nonzero(as_tuple=True)[0].tolist()
        for pos in mask_positions:
            if pos + 1 < SEQ_LEN and x_masked[pos + 1] == MASK_IDX:
                found_adjacent = True
            if pos - 1 >= 0 and x_masked[pos - 1] == MASK_IDX:
                found_adjacent = True
        if found_adjacent:
            break

    assert found_adjacent, "Adjacent masking for RNA was never triggered."


# ---------------------------------------------------------------------------
# Edge case: n_to_mask == 0
# ---------------------------------------------------------------------------


def test_zero_masking_rate_returns_zeros_in_y_masked():
    """When masked_rate rounds to 0 tokens, y_masked should be all zeros."""
    seq = [[1]]  # single-token sequence
    ds = MaskedDataset(x=seq, y=seq, max_len=1, mask_idx=99, masked_rate=0.01)
    x_masked, y_masked, x, _ = ds[0]
    assert (y_masked == 0).all()
    assert torch.equal(x_masked, x)
