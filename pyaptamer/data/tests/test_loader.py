"""Molecule data loading module tests."""

from pathlib import Path

from pyaptamer.data.loader import MoleculeLoader

# dataset paths relative to pyaptamer root
DATA_PATHS = [
    "datasets/data/pfoa.pdb",
    "datasets/data/1gnh.pdb",
]


def test_string_path():
    """Test loading with string path."""
    # get full paths first
    root_path = Path(__file__).parent.parent.parent
    full_paths = [str(root_path / p) for p in DATA_PATHS]

    loader = MoleculeLoader(full_paths)
    assert isinstance(loader, MoleculeLoader)

    pd_df = loader.to_df_seq()
    assert pd_df.shape == (len(DATA_PATHS), 1)

    one_gnh_str = pd_df.iloc[1, 0]
    assert one_gnh_str.startswith("QTDMSRK")


def test_pathlib_path():
    """Test loading with pathlib Path."""
    # get full paths first
    root_path = Path(__file__).parent.parent.parent
    full_paths = [root_path / p for p in DATA_PATHS]

    loader = MoleculeLoader(full_paths)
    assert isinstance(loader, MoleculeLoader)

    pd_df = loader.to_df_seq()
    assert pd_df.shape == (len(DATA_PATHS), 1)

    one_gnh_str = pd_df.iloc[1, 0]
    assert one_gnh_str.startswith("QTDMSRK")
