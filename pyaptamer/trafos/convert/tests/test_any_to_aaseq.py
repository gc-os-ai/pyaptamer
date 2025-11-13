"""
Unit tests for the AnyToAASeq transformer.
"""

import os

import pandas as pd
import pytest

from pyaptamer.trafos.convert import AnyToAASeq
from pyaptamer.utils import hf_to_dataset  # adjust import path if needed


@pytest.fixture
def fasta_file(tmp_path):
    """Download a small dataset from Hugging Face and save as a FASTA file."""
    ds = hf_to_dataset(
        "https://huggingface.co/datasets/gcos/HoloRBP4_round8_trimmed/resolve/main/HoloRBP4_round8_trimmed.fasta"
    )
    content = "\n".join(ds[:]["text"])
    fasta_path = tmp_path / "test_sequences.fasta"
    fasta_path.write_text(content)
    return fasta_path


@pytest.fixture
def pdb_file():
    """Return the path to the PDB test file (relative to this repo)."""
    pdb_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "datasets", "data", "1gnh.pdb"
    )
    pdb_path = os.path.abspath(pdb_path)
    return pdb_path


def test_fasta_to_aaseq(fasta_file):
    """Test converting FASTA input to amino acid sequences."""
    transformer = AnyToAASeq(format="fasta")
    df = transformer.transform(str(fasta_file))

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert "sequence" in df.columns
    assert len(df) > 0
    assert all(isinstance(seq, str) for seq in df["sequence"])
    assert df["sequence"].str.len().gt(0).all()


def test_pdb_to_aaseq(pdb_file):
    """Test converting a PDB input file to amino acid sequences."""
    transformer = AnyToAASeq(format="pdb-seqres")
    df = transformer.transform(pdb_file)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert "sequence" in df.columns
    assert len(df) > 0
    assert all(isinstance(seq, str) for seq in df["sequence"])
    assert df["sequence"].str.len().gt(0).all()
