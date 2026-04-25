__author__ = ["satvshr", "fkiraly"]
__all__ = ["load_pfoa"]

import os


def load_pfoa():
    """Load the PFOA molecule as a MoleculeLoader.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the PFOA molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "pfoa.pdb")

    return MoleculeLoader(pdb_path)
