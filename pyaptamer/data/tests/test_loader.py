"""Molecule data loading module tests."""

from pathlib import Path

import pytest

from pyaptamer.data.loader import MoleculeLoader

# dataset paths relative to pyaptamer root
VALID_DATA_PATHS = [
    "datasets/data/1gnh.pdb",
    "datasets/data/1brq.pdb",
    "datasets/data/5nu7.pdb",
]

INVALID_NO_SEQRES_PATH = "datasets/data/1gnh_no_seqres.pdb"


def test_string_path():
    """Test that MoleculeLoader works with string paths."""
    root_path = Path(__file__).parent.parent.parent
    full_paths = [str(root_path / p) for p in VALID_DATA_PATHS]

    loader = MoleculeLoader(full_paths)
    df = loader.to_df_seq()

    # column contract unchanged
    assert list(df.columns) == ["sequence"]

    # MultiIndex contract
    assert df.index.nlevels == 2
    assert df.index.names == ["path", "chain_id"]

    # all sequences are strings
    assert df["sequence"].map(type).eq(str).all()

    # at least one known sequence exists
    assert any(seq.startswith("QTDMSRK") for seq in df["sequence"])


def test_pathlib_path():
    """Test that MoleculeLoader works with pathlib.Path paths."""
    root_path = Path(__file__).parent.parent.parent
    full_paths = [root_path / p for p in VALID_DATA_PATHS]

    loader = MoleculeLoader(full_paths)
    df = loader.to_df_seq()

    assert list(df.columns) == ["sequence"]
    assert df.index.nlevels == 2
    assert df.index.names == ["path", "chain_id"]
    assert df["sequence"].map(type).eq(str).all()
    assert any(seq.startswith("QTDMSRK") for seq in df["sequence"])


def test_no_seqres_pdb_raises():
    """PDB files without SEQRES should raise."""
    root_path = Path(__file__).parent.parent.parent
    path = root_path / INVALID_NO_SEQRES_PATH

    loader = MoleculeLoader(path)

    with pytest.raises(ValueError, match="No sequences found"):
        loader.to_df_seq()


def test_ignore_duplicates_across_multiple_files():
    """Test ignore_duplicates filters duplicate sequences across files."""
    root_path = Path(__file__).parent.parent.parent
    # Use the same valid PDB file twice to simulate cross-file duplication
    path = root_path / "datasets/data/1brq.pdb"
    full_paths = [path, path]

    # Without ignore_duplicates, the sequences should be loaded twice
    loader_with_dups = MoleculeLoader(full_paths, ignore_duplicates=False)
    df_with_dups = loader_with_dups.to_df_seq()

    # With ignore_duplicates=True, duplicates from the second file
    # should be safely filtered out.
    loader_no_dups = MoleculeLoader(full_paths, ignore_duplicates=True)
    df_no_dups = loader_no_dups.to_df_seq()

    # Verify that the deduplicated DataFrame is strictly smaller than the duplicated one
    assert len(df_no_dups) < len(df_with_dups)
    # Since we passed the exact same file twice, it should be exactly half the size
    assert len(df_no_dups) == len(df_with_dups) // 2
