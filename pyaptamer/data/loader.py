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
    The loader determines the type from the file extension by default, but a
    format override can be provided to force a specific loader:

    - ``fmt="pdb"`` -> handled by :func:`pdb_to_aaseq` (PDB SEQRES extraction)
    - any other format string (e.g. ``"fasta"``, ``"genbank"``) -> passed to
      :mod:`Bio.SeqIO` for parsing

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
        Row index for the resulting DataFrame; if None, integer RangeIndex is used.
    columns : list, optional
        Column names for the resulting DataFrame; if None, defaults to ``["sequence"]``.
    fmt : str, optional
        Optional format override. If provided, this format will be used instead
        of inferring the format from the file extension. Examples: ``"pdb"``,
        ``"fasta"``, ``"genbank"``.
    """

    def __init__(self, path, index=None, columns=None, fmt=None):
        self.path = path
        self.index = index
        self.columns = columns
        self.fmt = fmt

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
        -------
        pd.DataFrame
            The sequence column contains:

            - for PDB files: a single amino-acid string,
              representing the concatenated unique chain sequences;
            - for SeqIO-readable formats: a list of amino-acid strings
              (one per sequence record in the file).

        """
        paths = self._path

        seq_list = [self._load_dispatch(path) for path in paths]

        if self.columns is None:
            columns = "sequence"
        else:
            columns = self.columns

        return pd.DataFrame({columns: seq_list}, index=self.index)

    def _determine_type(self, path):
        """Return file type inferred from suffix or from the instance `fmt` override.

        Parameters
        ----------
        path : Path
            Path to a file (used only when no fmt override is provided).

        Returns
        -------
        str
            Format string such as "pdb", "fasta", "genbank", etc.
        """
        # If user provided a format override, use it
        if self.fmt:
            return self.fmt

        # Fallback to suffix-based inference
        suffix = path.suffix.lower()
        if suffix == ".pdb":
            return "pdb"

        # strip leading dot; if no suffix, return None to indicate unknown
        return suffix.lstrip(".") if suffix else None

    def _load_dispatch(self, path):
        """Dispatch loader based on file type."""
        fmt = self._determine_type(path)

        if fmt == "pdb":
            return self._load_pdb_seq(path)

        if fmt is None:
            # no suffix and no format override -> error
            raise ValueError(
                f"Could not determine file format for '{path}'."
                "Provide a 'fmt' argument."
            )

        return self._load_seqio(path, fmt)

    def _load_pdb_seq(self, path):
        """Load a PDB file and extract its primary amino-acid sequence.

        Uses :func:`pdb_to_aaseq` internally.

        Parameters
        -----------
        path : Path
            path to PDB file

        Returns
        --------
        str
            Amino-acid sequences extracted from the PDB file.
        """
        sequence = pdb_to_aaseq(path)
        return "".join(sequence)

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
