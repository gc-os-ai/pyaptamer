"""Molecule data loading module."""

__all__ = ["MoleculeLoader"]
__author__ = ["fkiraly", "satvshr"]

from pathlib import Path

import pandas as pd
from Bio import SeqIO

from pyaptamer.utils import pdb_to_aaseq


class MoleculeLoader:
    """Molecule data connector.

    Parameters
    ------------
    path : str, Path, or list thereof
        file location or list of file locations with molecule files
        str can be any of the following... [fill in]

        the following formats are currently supported:

        * ``pdb`` format

    index : list, or pandas.Index coercible, optional
        row index for the structure; if None, integer RangeIndex is assumed

    columns : list, optional
        column names for the structure; if None, defaults to ["sequence"]
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
        """Return a pd.DataFrame of sequences.

        Returns
        --------
        pd.DataFrame
            string sequences in self in a pd.DataFrame of str
            has single column "sequence";
            rows are primary sequences found in the files in `path`
            sequences are determined from files as follows: [fill in]
        """
        paths = self._path

        seq_list = [self._load_dispatch(path) for path in paths]

        if self.columns is None:
            columns = ["sequence"]
        else:
            columns = self.columns

        return pd.DataFrame(seq_list, columns=columns, index=self.index)

    def _determine_type(self, path):
        suffix = path.suffix.lower()
        if suffix == ".pdb":
            return "pdb"

        # All other suffixes are handled by SeqIO
        return suffix[1:]

    def _load_dispatch(self, path):
        fmt = self._determine_type(path)

        if fmt == "pdb":
            return self._load_pdb_seq(path)

        return self._load_seqio(path, fmt)

    def _load_pdb_seq(self, path):
        """Load a PDB file and extract the primary sequence.

        Parameters
        -----------
        path : Path
            path to PDB file

        Returns
        --------
        list of str
            Sequences extracted from PDB files
        """
        sequence = pdb_to_aaseq(path)
        return sequence

    def _load_seqio(self, path, format):
        """Load any SeqIO-supported file format."""
        seqs = [str(rec.seq) for rec in SeqIO.parse(str(path), format)]
        if not seqs:
            raise ValueError(f"No sequences found in {path}")

        return seqs
