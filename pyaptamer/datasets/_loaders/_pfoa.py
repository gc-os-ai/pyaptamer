from __future__ import annotations

__author__ = ["satvshr", "fkiraly"]
__all__ = ["load_pfoa"]

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyaptamer.data.loader import MoleculeLoader


def load_pfoa() -> MoleculeLoader:
    """Load the PFOA molecule as a MoleculeLoader.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the PFOA molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "pfoa.pdb")

    return MoleculeLoader(pdb_path)
