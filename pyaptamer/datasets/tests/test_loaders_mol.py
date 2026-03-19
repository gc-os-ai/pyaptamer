"""Tests for molecule data loading module, toy data."""

from pathlib import Path

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
    assert df.shape == (10, 1)
    seq = df.iloc[0, 0]
    assert seq.startswith("QTDMSRK")


def test_loader_seqio_fasta_to_df_seq(tmp_path):
    """Test that SeqIO-backed formats are loaded into the same dataframe shape."""
    fasta_path = tmp_path / "toy.fasta"
    fasta_path.write_text(">seqA\nMKTAYIAKQRQISFVKSHFSRQ\n>seqB\nGILGYTEHQVVSSDFNSD\n")

    mol = MoleculeLoader(fasta_path)
    df = mol.to_df_seq()

    assert df.shape == (2, 1)
    assert df.index.names == ["path", "chain_id"]

    assert (Path(fasta_path), "seqA") in df.index
    assert (Path(fasta_path), "seqB") in df.index

    assert df.loc[(Path(fasta_path), "seqA"), "sequence"] == "MKTAYIAKQRQISFVKSHFSRQ"
    assert df.loc[(Path(fasta_path), "seqB"), "sequence"] == "GILGYTEHQVVSSDFNSD"
