__author__ = "satvshr"
__all__ = ["pdb_to_aaseq"]

import os

import pandas as pd
from Bio import SeqIO


def pdb_to_aaseq(pdb_file_path, return_type="list"):
    """
    Extract amino-acid sequences (SEQRES) from a PDB file.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
        Path to a PDB file.
    return_type : {'list', 'pd.df'}, optional, default='list'
        Format of the returned value:

        - ``'list'`` : return a Python list of sequence strings (one per chain).
        - ``'pd.df'`` : return a pandas.DataFrame indexed by chain id with a single
        column ``'sequence'`` containing one-letter amino-acid strings.

    Returns
    -------
    list of str or pandas.DataFrame
        Depending on ``return_type``. If ``'list'``, returns a list of sequence
        strings (one per SEQRES chain). If ``'pd.df'``, returns a DataFrame
        where the index is the chain identifier when present (index name ``'chain'``)
        and the column ``'sequence'`` contains the sequences. If no SEQRES records
        are found, returns an empty list or empty DataFrame respectively.
    """
    pdb_path = os.fspath(pdb_file_path)
    sequences = []
    chains = []

    with open(pdb_path) as handle:
        for record in SeqIO.parse(handle, "pdb-seqres"):
            sequences.append(str(record.seq))
            chains.append(getattr(record, "id", None))

    if return_type == "list":
        return sequences

    if return_type == "pd.df":
        df = pd.DataFrame({"chain": chains, "sequence": sequences})
        df = df.set_index("chain")
        return df
