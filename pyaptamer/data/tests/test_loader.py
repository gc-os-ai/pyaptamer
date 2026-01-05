"""Molecule data loading module tests."""

from pathlib import Path

from pyaptamer.data.loader import MoleculeLoader

# dataset paths relative to pyaptamer root
DATA_PATHS = [
    "datasets/data/1gnh_no_seqres.pdb",
    "datasets/data/1gnh.pdb",
]


def test_string_path():
    """Test loading with string path."""
    root_path = Path(__file__).parent.parent.parent
    full_paths = [str(root_path / p) for p in DATA_PATHS]

    loader = MoleculeLoader(full_paths)
    assert isinstance(loader, MoleculeLoader)

    pd_df = loader.to_df_seq()

    # one column, many rows (flattened sequences)
    assert pd_df.shape[1] == 1
    assert pd_df.shape[0] >= len(DATA_PATHS)

    # ensure sequences are strings, not lists
    assert isinstance(pd_df.iloc[0, 0], str)

    # check expected sequence content exists
    assert any(seq.startswith("QTDMSRK") for seq in pd_df["sequence"])


def test_pathlib_path():
    """Test loading with pathlib Path."""
    root_path = Path(__file__).parent.parent.parent
    full_paths = [root_path / p for p in DATA_PATHS]

    loader = MoleculeLoader(full_paths)
    assert isinstance(loader, MoleculeLoader)

    pd_df = loader.to_df_seq()

    # one column, many rows (flattened sequences)
    assert pd_df.shape[1] == 1
    assert pd_df.shape[0] >= len(DATA_PATHS)

    # ensure sequences are strings
    assert isinstance(pd_df.iloc[0, 0], str)

    # check expected sequence content exists
    assert any(seq.startswith("QTDMSRK") for seq in pd_df["sequence"])
