import os

from Bio.PDB import PDBParser


def load_1ghn_structure(pdb_path=None):
    """
    Load the 1ghn molecule from a PDB file using Biopython.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the PDB file. If not provided, the function uses the default path
        located in the '../data/1ghn.pdb' relative to the current file.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object representing the PFOA molecule.
    """
    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "1ghn.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1ghn", pdb_path)
    return structure
