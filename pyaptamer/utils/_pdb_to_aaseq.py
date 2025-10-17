__author__ = "satvshr"
__all__ = ["pdb_to_aaseq"]

import os

import pandas as pd
from Bio import SeqIO

from pyaptamer.utils._pdb_to_struct import pdb_to_struct
from pyaptamer.utils._struct_to_aaseq import struct_to_aaseq


def pdb_to_aaseq(pdb_file_path, return_type="list"):
    """
    Extract amino-acid sequences from a PDB file. Tries SEQRES records first
    (full deposited sequence). Falls back to using the package's
    pdb -> Structure -> sequences converters if SEQRES records are not present.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
        Path to a PDB file.
    return_type : {'list', 'pd.df'}, optional, default='list'
        Format of returned value:
          - ``'list'`` : Python list of amino-acid strings (one per chain / polypeptide)
          - ``'pd.df'`` : pandas.DataFrame with columns ``'chain'`` and ``'sequence'``
            (one row per chain/polypeptide). If chain ids are not available they may
            be ``None``.

    Returns
    -------
    list of str or pandas.DataFrame
        Depending on ``return_type``. If ``'list'``, returns a Python list of
        sequence strings (one element per chain / polypeptide). If ``'pd.df'``,
        returns a DataFrame with columns ``'chain'`` and ``'sequence'``.

    Raises
    ------
    FileNotFoundError
        If the given ``pdb_file_path`` does not exist.
    ValueError
        If ``return_type`` is invalid or no sequences could be extracted.
    """
    pdb_path = os.fspath(pdb_file_path)

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

    if return_type == "list":
        return df["sequence"].tolist()
    elif return_type == "pd.df":
        return df.reset_index(drop=True)[["chain", "sequence"]]
    else:
        raise ValueError("`return_type` must be either 'list' or 'pd.df'")
