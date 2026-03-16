"""Tests for molecule data loading module, toy data."""

import pytest

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import load_1brq, load_1gnh, load_5nu7, load_pfoa

LOADERS = [
    load_1gnh,
    load_5nu7,
    load_1brq,
    load_pfoa,
]


@pytest.mark.parametrize("loader", LOADERS)
def test_loader_returns_molecule_loader(loader):
    """Each loader should return a MoleculeLoader instance."""
    mol = loader()
    assert isinstance(mol, MoleculeLoader)


def test_loader_mol_to_df_seq():
    """Test that loader's to_df_seq method works correctly."""
    mol = load_1gnh()
    df = mol.to_df_seq()
    assert df.shape == (1, 1)
    seq = df.iloc[0, 0]
    assert seq.startswith("QTDMSRK")
