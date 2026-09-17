__author__ = ["satvshr", "siddharth7113"]
__all__ = ["load_from_rcsb"]

import os

from Bio.PDB import PDBList


def _default_pdb_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "pyaptamer", "pdb")


def load_from_rcsb(pdb_id, pdir=None, overwrite=False, tiling="bag"):
    """
    Download a PDB file from the RCSB Protein Data Bank
    and parse it into a `MoleculeLoader`.

    Parameters
    ----------
    pdb_id : str
        The 4-character PDB ID of the structure to download.

    pdir : str or os.PathLike, optional
        Directory the file is stored in as ``<pdb_id>.pdb``. Defaults to
        ``~/.cache/pyaptamer/pdb``. The file is reused on later calls.

    overwrite : bool, optional
        If True, download again even if the file already exists.
        Default is False.

    tiling : str, default="bag"
        Layout of the chains, passed to
        :class:`~pyaptamer.data.loader.MoleculeLoader`. ``"bag"`` keeps all
        chains in one cell, ``"samples"`` gives one row per chain.

    Returns
    -------
    MoleculeLoader
        A `MoleculeLoader` object for the downloaded structure.
    """
    from pyaptamer.data.loader import MoleculeLoader

    pdb_id = pdb_id.lower()
    pdir = os.fspath(pdir) if pdir is not None else _default_pdb_dir()
    os.makedirs(pdir, exist_ok=True)
    pdb_path = os.path.join(pdir, f"{pdb_id}.pdb")

    if overwrite or not os.path.exists(pdb_path):
        ent_path = PDBList(verbose=False).retrieve_pdb_file(
            pdb_id, pdir=pdir, file_format="pdb", overwrite=True
        )
        if not os.path.exists(ent_path):
            raise FileNotFoundError(f"Could not download PDB entry {pdb_id!r}")
        os.replace(ent_path, pdb_path)

    return MoleculeLoader(data={"sequence": [pdb_path]}, tiling=tiling)
