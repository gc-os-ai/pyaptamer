__author__ = "satvshr"
__all__ = ["load_from_rcsb"]

from Bio.PDB import PDBList

from pyaptamer.utils.pdb_to_struct import pdb_to_struct


def load_from_rcsb(pdb_id, overwrite=False):
    """
    Download a PDB file from the RCSB Protein Data Bank and parse it into a `Structure`.
    Files are created in the current working directory.

    Parameters
    ----------
    pdb_id : str
        The 4-character PDB ID of the structure to download.

    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object.
    """
    pdbl = PDBList()
    pdb_file_path = pdbl.retrieve_pdb_file(
        pdb_id, file_format="pdb", overwrite=overwrite
    )

    structure = pdb_to_struct(pdb_file_path)

    return structure
