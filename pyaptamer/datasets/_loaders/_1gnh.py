__author__ = ["satvshr", "fkiraly", "siddhart7113"]
__all__ = ["load_1gnh"]

import os


def load_1gnh(tiling="bag"):
    """Load the 1GNH molecule as a MoleculeLoader.

    Parameters
    ----------
    tiling : str, default="bag"
        Layout of the chains, passed to
        :class:`~pyaptamer.data.loader.MoleculeLoader`. ``"bag"`` keeps all
        chains in one cell, ``"samples"`` gives one row per chain.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the 1GNH molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "1gnh.pdb")

    return MoleculeLoader(data={"sequence": [pdb_path]}, tiling=tiling)
