"""Tests for torch transforms."""

import pytest
import torch

from pyaptamer.trafos.torch import (
    GreedyEncode,
    RandomMask,
)


class TestGreedyEncode:
    @pytest.fixture
    def vocab(self):
        return {"A": 1, "C": 2, "G": 3, "T": 4, "AC": 5, "GT": 6}

    def test_encode_simple(self, vocab):
        t = GreedyEncode(vocab, max_len=5)
        result = t("ACGT")
        assert result.shape == (5,)
        assert result[0].item() == 5
        assert result[1].item() == 6

    def test_padding(self, vocab):
        t = GreedyEncode(vocab, max_len=10)
        result = t("A")
        assert result.shape == (10,)
        assert result[0].item() == 1
        assert result[1].item() == 0

    def test_truncation(self, vocab):
        t = GreedyEncode(vocab, max_len=2)
        result = t("ACGTACGT")
        assert result.shape == (2,)

    def test_unknown_char(self, vocab):
        t = GreedyEncode(vocab, max_len=5)
        result = t("XYZ")
        assert result[0].item() == 0

    def test_repr(self, vocab):
        t = GreedyEncode(vocab, max_len=10)
        assert "GreedyEncode" in repr(t)


class TestRandomMask:
    def test_shape(self):
        t = RandomMask(mask_idx=99, mask_rate=0.5)
        x = torch.tensor([1, 2, 3, 4, 0, 0])
        assert t(x).shape == x.shape

    def test_preserves_padding(self):
        t = RandomMask(mask_idx=99, mask_rate=1.0)
        x = torch.tensor([1, 2, 0, 0])
        result = t(x)
        assert result[2].item() == 0
        assert result[3].item() == 0

    def test_applies_mask(self):
        torch.manual_seed(42)
        t = RandomMask(mask_idx=99, mask_rate=0.5)
        x = torch.tensor([1, 2, 3, 4])
        assert (t(x) == 99).any()

    def test_repr(self):
        t = RandomMask(mask_idx=99)
        assert "RandomMask" in repr(t)


class TestChaining:
    def test_tensor_transforms(self):
        vocab = {"A": 1, "C": 2, "G": 3, "U": 4}
        encode = GreedyEncode(vocab, max_len=5)
        mask = RandomMask(mask_idx=99, mask_rate=0.5)
        result = mask(encode("ACGU"))
        assert result.shape == (5,)
