__author__ = "satvshr"
__all__ = ["pdb_to_struct"]

import os
from typing import Union

from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure


def pdb_to_struct(pdb_file_path: Union[str, os.PathLike]) -> Structure:
    """
    Parse a PDB file into a Biopython Structure object.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
        Path to a local PDB file.

    Returns
    -------
    Bio.PDB.Structure.Structure
        Parsed Biopython structure object.

    Raises
    ------
    FileNotFoundError
        If the specified PDB file does not exist.

    Examples
    --------
    >>> from pyaptamer.utils import pdb_to_struct
    >>> # Assuming 'protein.pdb' is a valid PDB file
    >>> structure = pdb_to_struct("protein.pdb")
    >>> print(structure.get_id())
    protein.pdb
    """
    parser = PDBParser(QUIET=True)

    # Ensure compatibility with both strings and pathlib.Path objects
    safe_path = os.fspath(pdb_file_path)
    structure_id = os.path.basename(safe_path)

    structure = parser.get_structure(structure_id, safe_path)

    return structure