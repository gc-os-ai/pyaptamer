__author__ = "satvshr"
__all__ = ["load_1brq"]

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
