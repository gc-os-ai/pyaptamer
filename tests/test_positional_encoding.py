"""Tests for the `PositionalEncoding` layer (top-level to avoid package import).

Placed at repository `tests/` so pytest does not import the `pyaptamer` package
during collection (which would trigger optional runtime dependencies).
"""

__author__ = ["nennomp"]

import importlib.util
import pathlib

import pytest
import torch


@pytest.fixture(scope="module")
def PositionalEncoding():
    """Load `PositionalEncoding` from source without importing package root.

    The fixture resolves the module file path relative to the repository root
    and executes it as an isolated module, keeping tests hermetic.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    fp = repo_root / "pyaptamer" / "aptatrans" / "layers" / "_encoder.py"
    spec = importlib.util.spec_from_file_location("aptatrans_encoder", str(fp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PositionalEncoding


class TestPositionalEncoding:
    """Behavioral tests for `PositionalEncoding` (odd and even `d_model`)."""

    @pytest.mark.parametrize("d_model", [1, 2, 3, 4, 5, 8, 13])
    def test_forward_shape_preserved(self, PositionalEncoding, d_model):
        """Forward pass preserves the input shape for both odd and even dims."""
        pe = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=64)
        x = torch.zeros(2, 20, d_model)
        y = pe(x)

        assert y.shape == x.shape
        assert hasattr(pe, "pe")
        assert tuple(pe.pe.shape) == (1, pe.max_len, d_model)
