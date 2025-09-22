__author__ = "satvshr"
__all__ = ["pdb_to_aaseq"]

import os

import pandas as pd
from Bio import SeqIO


def pdb_to_aaseq(pdb_file_path, return_df=False):
    """
    Extract amino-acid sequences (SEQRES) from a PDB file.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
        Path to a PDB file.
    return_df : bool, optional, default=False
        If True, return a pandas.DataFrame with columns:
          - 'chain' (if available) and
          - 'sequence' (one-letter amino-acid string per chain).
        If False, return a list of strings (one per chain).

    Returns
    -------
    list of str or pandas.DataFrame
        List of amino-acid sequences (one-letter codes), or a DataFrame containing the
        sequences.extracted from the SEQRES records in the PDB file. Each element is a
        string representing the full sequence for a chain (e.g. "MKWVTFISLL..."). The
        order of sequences matches the order in which SEQRES records are encountered in
        the file. Returns an empty list if no SEQRES records are found.
    """
    pdb_path = os.fspath(pdb_file_path)
    sequences = []

    with open(pdb_path) as handle:
        for record in SeqIO.parse(handle, "pdb-seqres"):
            sequences.append(str(record.seq))

    if return_df:
        return pd.DataFrame({"sequence": sequences})

    return sequences
