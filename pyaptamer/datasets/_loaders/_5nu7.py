__author__ = "satvshr"
__all__ = ["load_5nu7"]

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
