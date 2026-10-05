__author__ = ["satvshr", "siddharth7113"]
__all__ = ["load_5nu7"]

import os


def load_5nu7(tiling="bag"):
    """Load the 5nu7 molecule as a MoleculeLoader.

    Parameters
    ----------
    tiling : str, default="bag"
        Layout of the chains, passed to
        :class:`~pyaptamer.data.loader.MoleculeLoader`. ``"bag"`` keeps all
        chains in one cell, ``"samples"`` gives one row per chain.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the 5nu7 molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "5nu7.pdb")

    return MoleculeLoader(data={"sequence": [pdb_path]}, tiling=tiling)
