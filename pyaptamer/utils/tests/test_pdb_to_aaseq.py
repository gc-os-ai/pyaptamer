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
def pdb_path_pfoa():
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "pfoa.pdb"
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

    assert not df.empty, "Returned DataFrame should not be empty"
    assert "sequence" in df.columns, "DataFrame should have a 'sequence' column"
    assert all(isinstance(s, str) and len(s) > 0 for s in df["sequence"]), (
        "Each sequence entry in DataFrame should be a non-empty string"
    )


def test_pdb_to_aaseq_atom_fallback(pdb_path_pfoa):
    """
    Use the packaged 'pfoa.pdb' (ATOM-only) to exercise the ATOM fallback.
    """

    sequences = pdb_to_aaseq(pdb_path_pfoa)
    print(sequences)
    assert isinstance(sequences, list), "Should return a list"
    assert len(sequences) > 0, "ATOM fallback should produce at least one sequence"


@pytest.mark.internet
def test_pdb_to_aaseq_uniprot_fetch(pdb_path_1gnh):
    """
    Test UniProt fetch mode using PDB ID 1gnh.
    (Requires internet connection)
    """
    try:
        sequences = pdb_to_aaseq(pdb_path_1gnh, use_uniprot=True, pdb_id="1gnh")
    except RuntimeError as e:
        pytest.skip(f"Skipped UniProt fetch test (API unavailable): {e}")
        return

    assert isinstance(sequences, list), "Expected list return type from UniProt mode"
    assert len(sequences) == 1, "Expected one canonical sequence"
    seq = sequences[0]
    assert isinstance(seq, str) and len(seq) > 50, (
        "Fetched UniProt sequence should be a non-trivial string"
    )
