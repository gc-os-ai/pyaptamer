__author__ = "fkiraly"
__all__ = ["load_1gnh"]

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
