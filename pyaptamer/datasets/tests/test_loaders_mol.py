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


def test_custom_columns(tmp_path):
    """Test that the columns parameter renames the output column."""
    fasta = tmp_path / "a.fasta"
    fasta.write_text(">seqA\nMKTAYIAKQRQISFVKSHFSRQ\n")

    mol = MoleculeLoader([fasta], columns=["seq"])
    df = mol.to_df_seq()

    assert df.columns.tolist() == ["seq"]
    assert df.iloc[0, 0] == "MKTAYIAKQRQISFVKSHFSRQ"


def test_custom_index(tmp_path):
    """Test that the index parameter replaces the default MultiIndex."""
    fasta = tmp_path / "a.fasta"
    fasta.write_text(">seqA\nMKTAYIAKQRQISFVKSHFSRQ\n>seqB\nGILGYTEHQVVSSDFNSD\n")

    mol = MoleculeLoader([fasta], index=["row0", "row1"])
    df = mol.to_df_seq()

    assert list(df.index) == ["row0", "row1"]
    assert df.loc["row0", "sequence"] == "MKTAYIAKQRQISFVKSHFSRQ"
    assert df.loc["row1", "sequence"] == "GILGYTEHQVVSSDFNSD"


def test_invalid_path_type_raises():
    """path must be str, Path, or list of those — anything else raises TypeError."""
    with pytest.raises(TypeError, match="path must be a str, Path, or list"):
        MoleculeLoader(42)


def test_fmt_override_no_suffix(tmp_path):
    """fmt override lets suffix-less files be parsed."""
    p = tmp_path / "mysequence"  # no suffix
    p.write_text(">seqA\nMKTAYIAKQRQISFVKSHFSRQ\n")

    mol = MoleculeLoader(p, fmt="fasta")
    df = mol.to_df_seq()

    assert df.shape == (1, 1)
    assert df.iloc[0, 0] == "MKTAYIAKQRQISFVKSHFSRQ"


def test_no_suffix_no_fmt_raises(tmp_path):
    """Suffix-less path with no fmt override raises ValueError."""
    p = tmp_path / "mysequence"
    p.write_text(">seqA\nMKTAYIAKQRQISFVKSHFSRQ\n")

    mol = MoleculeLoader(p)
    with pytest.raises(ValueError, match="Could not determine file format"):
        mol.to_df_seq()


def test_ignore_duplicates_with_custom_index(tmp_path):
    """Custom index is applied after dedup; its length must match kept rows."""
    fasta = tmp_path / "a.fasta"
    fasta.write_text(
        ">chainA\nAAA\n"
        ">chainB\nAAA\n"  # duplicate of chainA, removed by dedup
        ">chainC\nBBB\n"
    )
    # post-dedup: 2 rows (AAA, BBB), so index must be length 2
    mol = MoleculeLoader([fasta], index=["row0", "row1"], ignore_duplicates=True)
    df = mol.to_df_seq()

    assert list(df.index) == ["row0", "row1"]
    assert df.iloc[0, 0] == "AAA"
    assert df.iloc[1, 0] == "BBB"


def test_custom_index_multi_file(tmp_path):
    """Custom index replaces the multiindex even when rows span multiple files."""
    fasta_a = tmp_path / "a.fasta"
    fasta_a.write_text(">seqA\nAAA\n>seqB\nBBB\n")

    fasta_b = tmp_path / "b.fasta"
    fasta_b.write_text(">seqC\nCCC\n")

    mol = MoleculeLoader([fasta_a, fasta_b], index=["r0", "r1", "r2"])
    df = mol.to_df_seq()

    assert list(df.index) == ["r0", "r1", "r2"]
    assert df.loc["r0", "sequence"] == "AAA"
    assert df.loc["r2", "sequence"] == "CCC"


def test_custom_columns_multi_file(tmp_path):
    """Custom columns rename works across multiple files."""
    fasta_a = tmp_path / "a.fasta"
    fasta_a.write_text(">seqA\nAAA\n")

    fasta_b = tmp_path / "b.fasta"
    fasta_b.write_text(">seqB\nBBB\n")

    mol = MoleculeLoader([fasta_a, fasta_b], columns=["seq"])
    df = mol.to_df_seq()

    assert df.columns.tolist() == ["seq"]
    assert df.shape == (2, 1)


def test_ignore_duplicates(tmp_path):
    """Test that ignore_duplicates removes per-file duplicate sequences."""
    # file with two chains that share the same sequence
    fasta_a = tmp_path / "a.fasta"
    fasta_a.write_text(
        ">chainA\nMKTAYIAKQRQISFVKSHFSRQ\n"
        ">chainB\nMKTAYIAKQRQISFVKSHFSRQ\n"
        ">chainC\nGILGYTEHQVVSSDFNSD\n"
    )
    # second file has its own duplicate — should be deduped independently
    fasta_b = tmp_path / "b.fasta"
    fasta_b.write_text(
        ">chainD\nMKTAYIAKQRQISFVKSHFSRQ\n>chainE\nMKTAYIAKQRQISFVKSHFSRQ\n"
    )

    mol = MoleculeLoader([fasta_a, fasta_b], ignore_duplicates=True)
    df = mol.to_df_seq()

    # file a: chainA + chainC kept (chainB is a dup of chainA)
    # file b: chainD kept (chainE is a dup of chainD)
    assert len(df) == 3
    chains = df.index.get_level_values("chain_id").tolist()
    assert chains == ["chainA", "chainC", "chainD"]
