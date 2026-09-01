"""Tests for the masked language modeling dataset."""

__author__ = ["nennomp"]

import torch

from pyaptamer.datasets.dataclasses import MaskedDataset
from pyaptamer.datasets.dataclasses import _masked as masked_module


def test_masked_dataset_random_replacement_excludes_padding_tokens(monkeypatch):
    """Random replacement must not draw padding tokens from the vocabulary."""
    x = [list(range(1, 20)) + [0]]
    y = [list(range(1, 20)) + [0]]
    dataset = MaskedDataset(x, y, max_len=20, mask_idx=99, masked_rate=1.0)

    captured = {}

    def fake_sample(seq, k):
        return list(seq)[:k]

    def fake_choice(seq):
        captured["vocab"] = list(seq)
        return seq[0]

    monkeypatch.setattr(masked_module.random, "sample", fake_sample)
    monkeypatch.setattr(masked_module.random, "choice", fake_choice)

    x_masked, y_masked, x_orig, y_orig = dataset[0]

    assert 0 not in captured["vocab"]
    assert x_masked[0].item() == 99
    assert x_masked[14].item() == 99
    assert x_masked[15].item() == 1
    assert x_masked[16].item() == 17
    assert y_masked[15].item() == 16
    assert y_masked[19].item() == 0
    assert torch.equal(x_orig, torch.tensor(x[0], dtype=torch.int64))
    assert torch.equal(y_orig, torch.tensor(y[0], dtype=torch.int64))
