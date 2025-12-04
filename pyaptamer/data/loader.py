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
    """

    def __init__(self, path, index=None, columns=None, remove_duplicates=False):
        self.path = path
        self.index = index
        self.columns = columns
        self.remove_duplicates = remove_duplicates

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
            sequences in self in a pd.DataFrame;
            has single column "sequence";
            each row contains a list of str representing
            the primary sequence(s) found in the files in `path`
            sequences are determined from files as follows: [fill in]
        """
        paths = self._path

        # wrap each returned list so that it's a single DataFrame cell
        seq_list = [[self._load_dispatch(path, "seq")] for path in paths]

        if self.columns is None:
            columns = ["sequence"]
        else:
            columns = self.columns

        return pd.DataFrame(seq_list, columns=columns, index=self.index)

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
        return pdb_to_aaseq(path, remove_duplicates=self.remove_duplicates)
