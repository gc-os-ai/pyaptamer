"""Molecule data loading module."""

__all__ = ["MoleculeLoader"]
__author__ = ["fkiraly", "satvshr"]

from pathlib import Path

import pandas as pd
from Bio import SeqIO

from pyaptamer.utils import pdb_to_aaseq


class MoleculeLoader:
    """Molecule data connector.

    This loader extracts primary amino-acid sequences from molecule files.

    Supported file types
    --------------------
    The loader determines the type from the file extension:

    - ``.pdb`` -> handled by :func:`pdb_to_aaseq` (PDB SEQRES extraction)
    - any other extension -> passed to :mod:`Bio.SeqIO` for parsing

    Thus, all formats supported by :func:`Bio.SeqIO.parse` are accepted,
    including FASTA, GenBank, EMBL, FASTQ, and many more.

    For the full list of supported formats, see:

    - https://biopython.org/docs/latest/api/Bio.SeqIO.html#file-formats
    - https://biopython.org/wiki/SeqIO

    Parameters
    ----------
    path : str, Path, or list
        File location(s) of molecule files. One row is returned per file.
    index : list, or pandas.Index coercible, optional
        row index for the structure; if None, integer RangeIndex is assumed
    columns : list, optional
        Column names for the structure; if None, defaults to ``["sequence"]``.
    """

    def __init__(self, path, index=None, columns=None):
        self.path = path
        self.index = index
        self.columns = columns

        if isinstance(path, str):
            path = [Path(path)]
            self._path = path
        elif isinstance(path, Path):
            self._path = [path]
        elif isinstance(path, list):
            self._path = [Path(p) if isinstance(p, str) else p for p in path]

    def to_df_seq(self):
        """Return a DataFrame of amino-acid sequences.

        Each file in ``path`` yields one row in the output DataFrame.

        Returns
        --------
        pd.DataFrame
            The column ``"sequence"`` contains a list of sequences for each
            file. The list may contain:

            - one sequence (typical for PDB files),
            - or multiple sequences (e.g. multi-FASTA or multi-GenBank files).

        """
        paths = self._path

        seq_list = [self._load_dispatch(path) for path in paths]

        if self.columns is None:
            columns = ["sequence"]
        else:
            columns = self.columns

        return pd.DataFrame(seq_list, columns=columns, index=self.index)

    def _determine_type(self, path):
        """Return file type inferred from suffix."""
        suffix = path.suffix.lower()
        if suffix == ".pdb":
            return "pdb"

        # All other suffixes are treated as SeqIO formats
        return suffix[1:]

    def _load_dispatch(self, path):
        """Dispatch loader based on file type."""
        fmt = self._determine_type(path)

        if fmt == "pdb":
            return self._load_pdb_seq(path)

        return self._load_seqio(path, fmt)

    def _load_pdb_seq(self, path):
        """Load a PDB file and extract its amino-acid sequences.

        Uses :func:`pdb_to_aaseq` internally.

        Parameters
        -----------
        path : Path
            path to PDB file

        Returns
        --------
        list of str
            Amino-acid sequences extracted from PDB SEQRES records.
            Typically contains one sequence per chain.
        """
        sequence = pdb_to_aaseq(path)
        return sequence

    def _load_seqio(self, path, format):
        """Load any file format supported by Biopython SeqIO.

        Parameters
        ----------
        path : Path
            Path to a sequence file readable by SeqIO.
        format : str
            Biopython SeqIO format string (e.g. ``"fasta"``, ``"genbank"``).

        Returns
        -------
        list of str
            Amino-acid sequences extracted from the file.

        Raises
        ------
        ValueError
            If no sequences were found.
        """
        seqs = [str(rec.seq) for rec in SeqIO.parse(str(path), format)]
        if not seqs:
            raise ValueError(f"No sequences found in {path}")

        return seqs
