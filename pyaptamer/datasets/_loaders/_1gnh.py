from __future__ import annotations
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyaptamer.data.loader import MoleculeLoader
    import Bio.PDB.Structure


def load_1gnh(pdb_path: str | None = None) -> MoleculeLoader:
    """Load the 1GNH molecule as a MoleculeLoader.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the PDB file. If not provided, the function uses the default path
        located in the '../data/1gnh.pdb' relative to the current file.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the 1GNH molecule.
    """
    from pyaptamer.data.loader import MoleculeLoader

    if pdb_path is None:
        pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "1gnh.pdb")

    return MoleculeLoader(pdb_path)


# This function is provided only to test struct_to_aaseq.
def _load_1gnh_structure(pdb_path: str | None = None) -> Bio.PDB.Structure.Structure:
    """
    Load the 1gnh molecule from a PDB file using Biopython.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the PDB file. If not provided, the function uses the default path
        located in the '../data/1gnh.pdb' relative to the current file.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object representing the 1GNH molecule.
    """
    from Bio.PDB import PDBParser

    if pdb_path is None:
        pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "1gnh.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1gnh", pdb_path)
    return structure
