__author__ = "satvshr"
__all__ = ["pdb_to_struct"]

from Bio.PDB import PDBParser


def pdb_to_struct(pdb_file_path):
    """
    Parse a PDB file into a Biopython Structure object.

    Parameters
    ----------
    pdb_file_path : str
        Path to a local PDB file.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        Parsed Biopython structure object.
    """
    parser = PDBParser(QUIET=True)
    structure_id = pdb_file_path.split("/")[-1].split("\\")[-1]
    structure = parser.get_structure(structure_id, pdb_file_path)
    return structure
