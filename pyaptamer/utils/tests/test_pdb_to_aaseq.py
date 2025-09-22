__author__ = "satvshr"

import os

from pyaptamer.utils.pdb_to_aaseq import pdb_to_aaseq


def test_pdb_to_aaseq():
    """
    Test that `pdb_to_aaseq` converts a PDB file path into a non-empty string
    containing alphabetic characters.
    """
    pdb_file_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "1gnh.pdb"
    )
    sequences = pdb_to_aaseq(pdb_file_path)

    assert isinstance(sequences, list), "pdb_to_aaseq should return a list"
    assert len(sequences) > 0, "Returned list should not be empty"

    for seq in sequences:
        assert isinstance(seq, str), "Each entry should be a string"
        assert len(seq) > 0, "Each sequence string should not be empty"

    sequences = pdb_to_aaseq(pdb_file_path, return_df=True)

    assert not sequences.empty, "Returned DataFrame should not be empty"
    assert "sequence" in sequences.columns, "DataFrame should have a 'sequence' column"

    for seq in sequences["sequence"]:
        assert isinstance(seq, str), "Each entry should be a string"
        assert len(seq) > 0, "Each sequence string should not be empty"
