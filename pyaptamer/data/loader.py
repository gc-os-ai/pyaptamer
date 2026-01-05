"""Molecule data loading module."""

__all__ = ["MoleculeLoader"]
__author__ = ["fkiraly", "satvshr"]

from pathlib import Path

import pandas as pd

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
        """Return a pd.DataFrame of sequences with MultiIndex (path, seq_id).

        Returns
        --------
        pd.DataFrame
            sequences in self in a pd.DataFrame;
            index is a MultiIndex (path, seq_id);
            has single column "sequence";
            each row contains one primary amino-acid sequence
        """
        paths = self._path

        index_tuples = []
        sequences = []

        for path in paths:
            seqs = self._load_dispatch(path, "seq")
            for i, seq in enumerate(seqs):
                index_tuples.append((path, i))
                sequences.append(seq)

        index = pd.MultiIndex.from_tuples(index_tuples, names=["path", "seq_id"])

        columns = ["sequence"] if self.columns is None else self.columns

        return pd.DataFrame(sequences, index=index, columns=columns)

    def _determine_type(self, path):
        suffix = path.suffix.lower()
        if suffix == ".pdb":
            return "pdb"
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _load_dispatch(self, path, mode="seq"):
        ptype = self._determine_type(path)
        loader = getattr(self, f"_load_{ptype}_{mode}")
        return loader(path)

    def _load_pdb_seq(self, path):
        """Load a PDB file and extract the amino-acid sequences.

        Parameters
        -----------
        path : Path
            path to PDB file

        Returns
        --------
        List[str]
            primary sequence extracted from the PDB file as a list of strings
        """
        return pdb_to_aaseq(path, ignore_duplicates=self.ignore_duplicates)
