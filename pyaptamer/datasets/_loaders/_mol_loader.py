__author__ = "satvshr"
__all__ = ["mol_loader"]
import os


def mol_loader(pdb_name):
    """Create a MoleculeLoader for a packaged PDB file.

    This convenience factory constructs a MoleculeLoader pointing to a PDB file
    packaged with the distribution under pyaptamer/datasets/data/{pdb_name}.pdb.

    Parameters
    ----------
    pdb_name : str
        Basename of the packaged PDB file (without the ".pdb" extension).

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object for the requested packaged PDB file.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{pdb_name}.pdb")

    return MoleculeLoader(pdb_path)
