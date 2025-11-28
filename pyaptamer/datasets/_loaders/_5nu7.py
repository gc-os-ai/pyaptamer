__author__ = "satvshr"
__all__ = ["load_5nu7", "load_5nu7_structure"]

import os


def load_5nu7():
    """Load the 5nu7 molecule as a MoleculeLoader.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the 5nu7 molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "5nu7.pdb")

    return MoleculeLoader(pdb_path)


def load_5nu7_structure(pdb_path=None):
    """
    Load the 5nu7 molecule from a PDB file using Biopython.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the PDB file. If not provided, the function uses the default path
        located in the '../data/5nu7.pdb' relative to the current file.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object representing the 5NU7 molecule.
    """
    from Bio.PDB import PDBParser

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "5nu7.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("5nu7", pdb_path)
    return structure
