from Bio.PDB.Structure import Structure

from pyaptamer.datasets._loaders import load_pfoa_structure


def test_pfoa_loader():
    """
    Test that the load_pfoa_structure function runs without error and returns a valid
    Structure object.

    Asserts
    -------
        The datasets loads and the return value must be an instance of Biopython's
        Structure class.
    """
    structure = load_pfoa_structure()

    assert isinstance(structure, Structure), (
        "Returned object is not a Biopython Structure"
    )
