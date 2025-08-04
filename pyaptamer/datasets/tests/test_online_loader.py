import pytest

from pyaptamer.datasets import download_and_extract_sequences


@pytest.mark.parametrize("pdb_id", "1GNH")
def test_download_and_extract_sequences(pdb_id):
    """
    The download_and_extract_sequences function works correctly
    for valid PDB IDs.
    """
    sequences = download_and_extract_sequences(pdb_id)
    assert isinstance(sequences, list), (
        f"Expected a list of sequences, got {type(sequences)}"
    )
