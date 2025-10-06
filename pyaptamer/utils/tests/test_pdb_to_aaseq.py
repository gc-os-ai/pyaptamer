__author__ = "satvshr"

import os

import pytest

from pyaptamer.utils import pdb_to_aaseq


@pytest.fixture
def pdb_path():
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "1gnh.pdb"
    )


def test_pdb_to_aaseq_seqres(pdb_path):
    """
    Test that pdb_to_aaseq correctly extracts SEQRES sequences as a list and DataFrame.
    """
    sequences = pdb_to_aaseq(pdb_path)

    assert isinstance(sequences, list), "Expected a list return type"
    assert len(sequences) > 0, "Returned list should not be empty"

    for seq in sequences:
        assert isinstance(seq, str), "Each entry should be a string"
        assert seq.isalpha(), "Sequence should contain only alphabetic characters"
        assert len(seq) > 0, "Sequence should not be empty"

    df = pdb_to_aaseq(pdb_path, return_type="pd.df")

    assert not df.empty, "Returned DataFrame should not be empty"
    assert "sequence" in df.columns, "DataFrame should have a 'sequence' column"
    assert all(isinstance(s, str) and len(s) > 0 for s in df["sequence"]), (
        "Each sequence entry in DataFrame should be a non-empty string"
    )


def test_pdb_to_aaseq_atom_fallback(tmp_path):
    """
    Test the ATOM fallback by creating a minimal PDB file without SEQRES records.
    """
    pdb_data = """\
ATOM      1  N   MET A   1      11.104  13.207  10.248  1.00 20.00           N
ATOM      2  CA  MET A   1      12.560  13.200  10.248  1.00 20.00           C
ATOM      3  C   MET A   1      13.026  11.770  10.448  1.00 20.00           C
ATOM      4  O   MET A   1      12.626  10.898   9.680  1.00 20.00           O
TER
END
"""
    test_pdb = tmp_path / "test_atom_only.pdb"
    test_pdb.write_text(pdb_data)

    sequences = pdb_to_aaseq(test_pdb)
    assert isinstance(sequences, list), "Should return a list"
    assert len(sequences) > 0, "ATOM fallback should produce a sequence"


@pytest.mark.internet
def test_pdb_to_aaseq_uniprot_fetch(monkeypatch):
    """
    Test UniProt fetch mode using PDB ID 1gnh.
    (Requires internet connection)
    """
    pdb_id = "1gnh"

    try:
        sequences = pdb_to_aaseq("dummy_path", use_uniprot=True, pdb_id=pdb_id)
    except RuntimeError as e:
        pytest.skip(f"Skipped UniProt fetch test (API unavailable): {e}")
        return

    assert isinstance(sequences, list), "Expected list return type from UniProt mode"
    assert len(sequences) == 1, "Expected one canonical sequence"
    seq = sequences[0]
    assert isinstance(seq, str) and len(seq) > 50, (
        "Fetched UniProt sequence should be a non-trivial string"
    )


def test_invalid_return_type(pdb_path):
    """
    Test that invalid return_type raises ValueError.
    """
    with pytest.raises(ValueError):
        pdb_to_aaseq(pdb_path, return_type="invalid")
