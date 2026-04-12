__author__ = "satvshr"
__all__ = ["load_from_rcsb"]

from pathlib import Path

from Bio.PDB import PDBList

from pyaptamer.data.loader import MoleculeLoader


def load_from_rcsb(pdb_id, overwrite=False):
    """
    Download a PDB file from the RCSB Protein Data Bank and load it as a MoleculeLoader.
    Files are created in the current working directory.

    Parameters
    ----------
    pdb_id : str
        The 4-character PDB ID of the structure to download.

    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the downloaded molecule.
    """
    pdbl = PDBList()
    pdb_file_path = Path(
        pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", overwrite=overwrite)
    )

    # BioPython saves PDB files with .ent extension; rename to .pdb
    # so MoleculeLoader can recognize the file type
    if pdb_file_path.suffix == ".ent":
        pdb_path = pdb_file_path.with_suffix(".pdb")
        pdb_file_path.replace(pdb_path)
        pdb_file_path = pdb_path

    return MoleculeLoader(pdb_file_path)
