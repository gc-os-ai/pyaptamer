import os

from Bio.PDB import PDBParser


def load_pfoa_structure(pdb_path=None):
    """
    Load the PFOA molecule from a PDB file using Biopython.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the PDB file. If not provided, the function uses the default path
        located in the '../data/pfoa.pdb' relative to the current file.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object representing the PFOA molecule.
    """
    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "pfoa.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PFOA", pdb_path)
    return structure
