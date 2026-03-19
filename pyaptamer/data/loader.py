"""Molecule data loading module."""

__all__ = ["MoleculeLoader"]
__author__ = ["fkiraly", "satvshr"]

from pathlib import Path

import pandas as pd
from Bio import SeqIO
<<<<<<< HEAD

from pyaptamer.utils import pdb_to_aaseq
=======
>>>>>>> origin/main


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
        column names for the structure; if None, defaults to ["sequence"]

    ignore_duplicates : bool, optional, default=False
        if True, removes duplicate sequences (keeping the first occurrence).
    """

    def __init__(self, path, index=None, columns=None, ignore_duplicates=False):
        self.path = path
        self.index = index
        self.columns = columns
        self.ignore_duplicates = ignore_duplicates
        if isinstance(path, str):
            path = [Path(path)]
            self._path = path
        elif isinstance(path, Path):
            self._path = [path]
        elif isinstance(path, list):
            self._path = [Path(p) if isinstance(p, str) else p for p in path]

    def to_df_seq(self):
        """Return a pd.DataFrame of sequences with MultiIndex (path, chain_id).

        Returns
        -------
        pd.DataFrame
            sequences in self in a pd.DataFrame;
            index is a MultiIndex (path, chain_id);
            has single column "sequence";
            each row contains one primary amino-acid sequence
        """
        paths = self._path

        index_tuples = []
        sequences = []

        for path in paths:
            seqs = self._load_dispatch(path, "seq")
            for _, row in seqs.iterrows():
                index_tuples.append((path, row["chain_id"]))
                sequences.append(row["sequence"])

        index = pd.MultiIndex.from_tuples(index_tuples, names=["path", "chain_id"])

        columns = ["sequence"] if self.columns is None else self.columns

        return pd.DataFrame(sequences, index=index, columns=columns)

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
        """Load a PDB file and extract the amino-acid sequences.

        Parameters
        -----------
        path : Path
            path to PDB file

        Returns
        --------
        pandas.DataFrame
            DataFrame with columns ``["chain_id", "sequence"]``.
        """
        with open(path) as handle:
            seqres_records = list(SeqIO.parse(handle, "pdb-seqres"))

        records = [
            {
                "chain_id": record.id.split(":")[1] if ":" in record.id else record.id,
                "sequence": str(record.seq),
            }
            for record in seqres_records
        ]
        return pd.DataFrame.from_records(records, columns=["chain_id", "sequence"])
