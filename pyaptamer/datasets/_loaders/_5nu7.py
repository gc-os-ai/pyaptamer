from __future__ import annotations
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyaptamer.data.loader import MoleculeLoader


def load_5nu7(pdb_path: str | None = None) -> MoleculeLoader:
    """Load the 5nu7 molecule as a MoleculeLoader.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the PDB file. If not provided, the function uses the default path
        located in the '../data/5nu7.pdb' relative to the current file.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the 5nu7 molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    if pdb_path is None:
        pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "5nu7.pdb")

    return MoleculeLoader(pdb_path)
