"""Molecule data loading module tests."""

from pathlib import Path

import pandas as pd
import pytest

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.utils import hf_to_dataset

# dataset paths relative to pyaptamer root
DATA_PATHS = [
    "datasets/data/1gnh.pdb",
    "datasets/data/1gnh_no_seqres.pdb",
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


@pytest.fixture
def fasta_file(tmp_path):
    """Download FASTA file from Hugging Face and save locally."""
    ds = hf_to_dataset(
        "https://huggingface.co/datasets/gcos/HoloRBP4_round8_trimmed/resolve/main/HoloRBP4_round8_trimmed.fasta"
    )
    content = "\n".join(ds[:]["text"])
    fasta_path = tmp_path / "test_sequences.fasta"
    fasta_path.write_text(content)
    return fasta_path


def test_fasta_loading(fasta_file):
    """Test loading amino-acid sequences from a FASTA file via MoleculeLoader."""
    loader = MoleculeLoader([fasta_file])

    df = loader.to_df_seq()

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 1)
    assert "sequence" in df.columns

    seqs = df.iloc[0, 0]

    assert isinstance(seqs, list)
    assert len(seqs) > 0
    assert all(isinstance(s, str) for s in seqs)
    assert all(len(s) > 0 for s in seqs)
