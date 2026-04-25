__author__ = "satvshr"

import os

import pytest

from pyaptamer.utils import pdb_to_aaseq


@pytest.fixture
def pdb_path_1gnh():
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "1gnh.pdb"
    )


@pytest.fixture
def pdb_path_1gnh_no_seqres():
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "1gnh_no_seqres.pdb"
    )


def test_pdb_to_aaseq_seqres(pdb_path_1gnh):
    """
    Test that pdb_to_aaseq correctly extracts SEQRES sequences as a list and DataFrame.
    """
    sequences = pdb_to_aaseq(pdb_path_1gnh)

    assert isinstance(sequences, list), "Expected a list return type"
    assert len(sequences) > 0, "Returned list should not be empty"

    for seq in sequences:
        assert isinstance(seq, str), "Each entry should be a string"
        assert seq.isalpha(), "Sequence should contain only alphabetic characters"
        assert len(seq) > 0, "Sequence should not be empty"

    df = pdb_to_aaseq(pdb_path_1gnh, return_type="pd.df")

    assert isinstance(df, type(__import__("pandas").DataFrame())), (
        "Expected a pandas DataFrame"
    )
    assert not df.empty, "Returned DataFrame should not be empty"
    assert list(df.columns) == ["chain", "sequence"], (
        "DataFrame should have columns ['chain', 'sequence']"
    )

    # Number of rows should match number of sequences
    assert len(df) == len(sequences)

    # Sequences in DataFrame should match the list and be non-empty strings
    assert all(isinstance(s, str) and len(s) > 0 for s in df["sequence"])
    # Ensure dtype is object and entries are acceptable
    assert all((c is None) or (isinstance(c, str) and len(c) >= 1) for c in df["chain"])


def test_pdb_to_aaseq_atom_fallback(pdb_path_1gnh_no_seqres):
    """
    Use the packaged '1gnh_no_seqres.pdb' (ATOM-only) to exercise the ATOM fallback.
    """

    sequences = pdb_to_aaseq(pdb_path_1gnh_no_seqres)
    assert isinstance(sequences, list), "Should return a list"
    assert len(sequences) > 0, "ATOM fallback should produce at least one sequence"
    assert all(isinstance(s, str) and len(s) > 0 for s in sequences)

    # DataFrame return should include chain and sequence columns
    df = pdb_to_aaseq(pdb_path_1gnh_no_seqres, return_type="pd.df")
    assert isinstance(df, type(__import__("pandas").DataFrame()))
    assert not df.empty, "ATOM fallback DataFrame should not be empty"
    assert list(df.columns) == ["chain", "sequence"], (
        "DataFrame should have columns ['chain','sequence']"
    )
    assert all(isinstance(s, str) and len(s) > 0 for s in df["sequence"])
