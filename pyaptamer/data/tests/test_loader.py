"""Molecule data loading module tests."""

from pathlib import Path

from pyaptamer.data.loader import MoleculeLoader

# dataset paths relative to pyaptamer root
DATA_PATHS = [
    "datasets/data/1gnh_no_seqres.pdb",
    "datasets/data/1gnh.pdb",
]


def test_string_path():
    """Test that MoleculeLoader works with string paths."""
    root_path = Path(__file__).parent.parent.parent
    full_paths = [str(root_path / p) for p in DATA_PATHS]

    loader = MoleculeLoader(full_paths)
    df = loader.to_df_seq()

    # column contract unchanged
    assert list(df.columns) == ["sequence"]

    # MultiIndex contract
    assert df.index.nlevels == 2
    assert df.index.names == ["path", "seq_id"]

    # all sequences are strings
    assert df["sequence"].map(type).eq(str).all()

    # at least one known sequence exists
    assert any(seq.startswith("QTDMSRK") for seq in df["sequence"])


def test_pathlib_path():
    """Test that MoleculeLoader works with pathlib.Path paths."""
    root_path = Path(__file__).parent.parent.parent
    full_paths = [root_path / p for p in DATA_PATHS]

    loader = MoleculeLoader(full_paths)
    df = loader.to_df_seq()

    assert list(df.columns) == ["sequence"]
    assert df.index.nlevels == 2
    assert df.index.names == ["path", "seq_id"]
    assert df["sequence"].map(type).eq(str).all()
    assert any(seq.startswith("QTDMSRK") for seq in df["sequence"])
