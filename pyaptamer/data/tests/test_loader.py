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
