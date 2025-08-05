import os

from pyaptamer.utils.pdb_to_struct import pdb_to_struct


def test_pdb_to_struct():
    """
    Test that `pdb_to_struct` correctly converts a PDB file path into a Biopython
    Structure object.
    """
    pdb_file_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "1gnh.pdb"
    )
    structure = pdb_to_struct(pdb_file_path)

    assert hasattr(structure, "get_atoms"), "Structure should have get_atoms method"
