import pytest
from Bio.PDB.Structure import Structure

from pyaptamer.datasets.loader import load_pfoa_structure


def test_load_pfoa_structure_runs_and_returns_structure():
    """
    Test that the load_pfoa_structure function runs without error and returns a valid
    Structure object.

    Asserts
    -------
        The datasets loads and the return value must be an instance of Biopython's
        Structure class.
    """
    try:
        structure = load_pfoa_structure()
    except Exception as e:
        pytest.fail(f"load_pfoa_structure raised an exception: {e}")

    assert isinstance(structure, Structure), (
        "Returned object is not a Biopython Structure"
    )
