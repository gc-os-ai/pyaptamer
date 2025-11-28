__author__ = "satvshr"
__all__ = ["load_1brq", "load_1brq_structure"]

import os


def load_1brq():
    """Load the 1brq molecule as a MoleculeLoader.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the 1brq molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "1brq.pdb")

    return MoleculeLoader(pdb_path)


def load_1brq_structure(pdb_path=None):
    """
    Load the 1brq molecule from a PDB file using Biopython.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the PDB file. If not provided, the function uses the default path
        located in the '../data/1brq.pdb' relative to the current file.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object representing the 1BRQ molecule.
    """
    from Bio.PDB import PDBParser

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "1brq.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1brq", pdb_path)
    return structure
