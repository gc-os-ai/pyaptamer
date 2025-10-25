"""Tests for molecule data loading module, toy data."""

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import load_1gnh, load_pfoa


def test_loaders_mol():
    """Placeholder test for molecule data loading module."""
    pfoa = load_pfoa()
    assert isinstance(pfoa, MoleculeLoader)

    gnh = load_1gnh()
    assert isinstance(gnh, MoleculeLoader)

    gnh_seq = gnh.to_df_seq().iloc[0, 0]
    assert gnh_seq.startswith("QTDMSRK")
