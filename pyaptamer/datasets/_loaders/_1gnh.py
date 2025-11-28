__author__ = ["satvshr", "fkiraly"]
__all__ = ["load_1gnh", "load_1gnh_structure"]

import os


def load_1gnh():
    """Load the 1GNH molecule as a MoleculeLoader.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the 1GNH molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "1gnh.pdb")

    return MoleculeLoader(pdb_path)


def load_1gnh_structure(pdb_path=None):
    """
    Load the 1gnh molecule from a PDB file using Biopython.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the PDB file. If not provided, the function uses the default path
        located in the '../data/1gnh.pdb' relative to the current file.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object representing the 1GNH molecule.
    """
    from Bio.PDB import PDBParser

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "1gnh.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1gnh", pdb_path)
    return structure
