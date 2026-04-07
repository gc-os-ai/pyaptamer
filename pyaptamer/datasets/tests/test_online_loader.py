__author__ = "satvshr"

from unittest.mock import patch

import pytest
from Bio.PDB.Structure import Structure

from pyaptamer.datasets import load_from_rcsb


@pytest.mark.parametrize("pdb_id", ["1GNH"])
def test_download_structure(pdb_id):
    """
    The download_structure function works correctly
    for valid PDB IDs.
    """
    structure = load_from_rcsb(pdb_id)
    assert isinstance(structure, Structure), (
        f"Expected a Bio.PDB.Structure.Structure, got {type(structure)}"
    )


def test_download_structure_raises_clear_error_when_download_fails():
    """A failed RCSB download raises a clear error before parsing."""
    with patch("pyaptamer.datasets._loaders._online_databank.PDBList") as mock_pdbl:
        mock_pdbl.return_value.retrieve_pdb_file.return_value = None

        with pytest.raises(FileNotFoundError) as exc_info:
            load_from_rcsb("XXXX")

    assert str(exc_info.value) == "Failed to download PDB structure 'XXXX'."
