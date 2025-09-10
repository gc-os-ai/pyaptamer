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
