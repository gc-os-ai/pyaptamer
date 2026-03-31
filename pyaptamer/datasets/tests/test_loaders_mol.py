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


def test_loader_seqio_fasta_to_df_seq_multiindex(tmp_path):
    """Test multiindex DataFrame from multiple fasta files."""
    fasta_a = tmp_path / "a.fasta"
    fasta_a.write_text(">seqA\nMKTAYIAKQRQISFVKSHFSRQ\n>seqB\nGILGYTEHQVVSSDFNSD\n")

    fasta_b = tmp_path / "b.fasta"
    fasta_b.write_text(">seqC\nMVLSPADKTNVKAAWGKVGA\n")

    mol = MoleculeLoader([fasta_a, fasta_b])
    df = mol.to_df_seq()

    # shape: 3 sequences across 2 files, 1 column
    assert df.shape == (3, 1)
    assert df.index.names == ["path", "chain_id"]
    assert df.columns.tolist() == ["sequence"]

    # both paths appear in the first index level
    paths = df.index.get_level_values("path")
    assert set(paths) == {Path(fasta_a), Path(fasta_b)}

    # sequences are correctly assigned
    assert df.loc[(Path(fasta_a), "seqA"), "sequence"] == "MKTAYIAKQRQISFVKSHFSRQ"
    assert df.loc[(Path(fasta_a), "seqB"), "sequence"] == "GILGYTEHQVVSSDFNSD"
    assert df.loc[(Path(fasta_b), "seqC"), "sequence"] == "MVLSPADKTNVKAAWGKVGA"
