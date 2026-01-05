"""Tests for molecule data loading module, toy data."""

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import load_1gnh, load_pfoa


def test_loaders_mol():
    """Placeholder test for molecule data loading module."""
    pfoa = load_pfoa()
    assert isinstance(pfoa, MoleculeLoader)

    gnh = load_1gnh()
    assert isinstance(gnh, MoleculeLoader)

    df = gnh.to_df_seq()
    assert any(seq.startswith("QTDMSRK") for seq in df["sequence"])
