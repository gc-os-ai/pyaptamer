__author__ = "satvshr"
__all__ = ["load_from_rcsb"]

import os

from Bio.PDB import PDBList


def load_from_rcsb(pdb_id, overwrite=False):
    """
    Download a PDB file from the RCSB Protein Data Bank
    and parse it into a `MoleculeLoader`.
    Files are created in the current working directory.

    Parameters
    ----------
    pdb_id : str
        The 4-character PDB ID of the structure to download.

    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    Returns
    -------
    MoleculeLoader
        A `MoleculeLoader` object for the downloaded structure.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdbl = PDBList()
    pdb_file_path = pdbl.retrieve_pdb_file(
        pdb_id, file_format="pdb", overwrite=overwrite
    )

    # BioPython saves PDB files with .ent extension; rename to .pdb
    # so MoleculeLoader can recognize the file type
    root, ext = os.path.splitext(pdb_file_path)
    if ext == ".ent":
        pdb_path = root + ".pdb"
        if overwrite or not os.path.exists(pdb_path):
            os.rename(pdb_file_path, pdb_path)
        pdb_file_path = pdb_path

    return MoleculeLoader(pdb_file_path)
