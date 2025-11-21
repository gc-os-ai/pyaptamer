import os

from Bio.PDB import PDBParser


def structure_loader(pdb_name):
    """
    Load a packaged PDB file from pyaptamer/datasets/data as a Biopython Structure.

    This loader only loads PDB files bundled with the package. It looks for
    the file "<package>/pyaptamer/datasets/data/{pdb_name}.pdb" and parses it
    with Bio.PDB.PDBParser.

    Parameters
    ----------
    pdb_name : str, optional
        Basename of the packaged PDB file (without the ".pdb" extension).

    Returns
    -------
    Bio.PDB.Structure.Structure
        Parsed Biopython Structure object.

    Raises
    ------
    FileNotFoundError
        If the requested packaged PDB file does not exist.
    """
    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{pdb_name}.pdb")
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Packaged PDB not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    return structure
