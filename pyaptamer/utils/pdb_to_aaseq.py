__author__ = "satvshr"
__all__ = ["pdb_to_aaseq"]

import os

from Bio import SeqIO


def pdb_to_aaseq(pdb_file_path):
    """
    Extract amino-acid sequences (SEQRES) from a PDB file.

    Parameters
    ----------
    pdb_file : str or os.PathLike
        Path to a PDB file.

    Returns
    -------
    list of str
        List of amino-acid sequences (one-letter codes) extracted from the SEQRES
        records in the PDB file. Each element is a string representing the full sequence
        for a chain (e.g. "MKWVTFISLL..."). The order of sequences matches the order in
        which SEQRES records are encountered in the file. Returns an empty list if no
        SEQRES records are found.
    """
    pdb_path = os.fspath(pdb_file_path)
    sequences = []
    with open(pdb_path) as handle:
        for record in SeqIO.parse(handle, "pdb-seqres"):
            sequences.append(str(record.seq))
    return sequences
