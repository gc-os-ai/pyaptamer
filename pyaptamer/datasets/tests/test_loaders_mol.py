"""Tests for molecule data loading module, toy data."""

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import mol_loader


def test_loaders_mol():
    """Placeholder test for molecule data loading module."""
    nu7 = mol_loader("5nu7")
    assert isinstance(nu7, MoleculeLoader)

    gnh = mol_loader("1gnh")
    assert isinstance(gnh, MoleculeLoader)

    gnh_seq = gnh.to_df_seq().iloc[0, 0]
    assert gnh_seq.startswith("QTDMSRK")
