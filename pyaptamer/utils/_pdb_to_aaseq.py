__author__ = "satvshr"
__all__ = ["pdb_to_aaseq"]

import os

import pandas as pd
from Bio import SeqIO

from pyaptamer.utils._pdb_to_struct import pdb_to_struct
from pyaptamer.utils._struct_to_aaseq import struct_to_aaseq


def pdb_to_aaseq(pdb_file_path, return_type="list", ignore_duplicates=False):
    """
    Extract amino-acid sequences from a PDB file.

    Tries SEQRES records first (full deposited sequence). Falls back to using
    the package's pdb -> Structure -> sequences converters if SEQRES records
    are not present.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
        Path to a PDB file.
    return_type : {'list', 'pd.df'}, optional, default='list'
        Format of returned value:
          - ``'list'`` : Python list of amino-acid strings (one per chain / polypeptide)
          - ``'pd.df'`` : pandas.DataFrame with columns ``'chain'`` and ``'sequence'``
            (one row per chain/polypeptide). Chain IDs may be ``None`` if unavailable.
    ignore_duplicates : bool, optional, default=False
        If True, removes duplicate sequences (keeping the first occurrence).
        Duplicates are identified by comparing the ``'sequence'`` column only.

    Returns
    -------
    list of str or pandas.DataFrame
        Depending on ``return_type``:
          - If ``'list'``: Python list of sequence strings (one per chain/polypeptide)
          - If ``'pd.df'``: DataFrame with columns ``['chain', 'sequence']``

    Raises
    ------
    TypeError
        If ``pdb_file_path`` is None or not a valid path type.
    FileNotFoundError
        If the given ``pdb_file_path`` does not exist.
    ValueError
        If ``return_type`` is invalid or no sequences could be extracted.
    """
    # Validate return_type early
    if return_type not in ["list", "pd.df"]:
        raise ValueError("`return_type` must be either 'list' or 'pd.df'")

    # Validate ignore_duplicates
    if not isinstance(ignore_duplicates, bool):
        raise TypeError(
            f"`ignore_duplicates` must be a boolean, "
            f"got {type(ignore_duplicates).__name__}"
        )

    # Validate pdb_file_path is not None
    if pdb_file_path is None:
        raise TypeError("`pdb_file_path` cannot be None")

    # Convert to path and validate
    try:
        pdb_path = os.fspath(pdb_file_path)
    except TypeError as err:
        raise TypeError(
            f"`pdb_file_path` must be a string or os.PathLike, "
            f"got {type(pdb_file_path).__name__}"
        ) from err

    # Check file existence before attempting to open
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Try SEQRES records first
    with open(pdb_path) as handle:
        seqres_records = list(SeqIO.parse(handle, "pdb-seqres"))

    if seqres_records:
        records = [
            {"chain": getattr(record, "id", None), "sequence": str(record.seq)}
            for record in seqres_records
        ]
        df = pd.DataFrame.from_records(records, columns=["chain", "sequence"])
    else:
        # Fall back to structure parsing
        structure = pdb_to_struct(pdb_path)
        df = struct_to_aaseq(structure, return_type="pd.df")

    if df.empty:
        raise ValueError(f"No sequences could be extracted from PDB file: {pdb_path}")

    # Remove duplicate sequences
    if ignore_duplicates:
        df = df.drop_duplicates(subset=["sequence"], keep="first").reset_index(
            drop=True
        )

    if return_type == "list":
        return df["sequence"].tolist()
    else:  # return_type == "pd.df"
        return df.reset_index(drop=True)[["chain", "sequence"]]
