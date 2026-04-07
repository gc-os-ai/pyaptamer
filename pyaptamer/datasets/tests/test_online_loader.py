__author__ = "satvshr"

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


def test_download_structure_raises_when_download_fails(monkeypatch):
    """Test FileNotFoundError raised when the PDB download fails."""

    monkeypatch.setattr(
        "pyaptamer.datasets._loaders._online_databank.PDBList.retrieve_pdb_file",
        lambda self, pdb_id, file_format="pdb", overwrite=False: None,
    )

    with pytest.raises(
        FileNotFoundError, match="Failed to download PDB structure 'XXXX'"
    ):
        load_from_rcsb("XXXX")
