"""Molecule data loading module."""

__all__ = ["MoleculeLoader"]
__author__ = ["fkiraly", "satvshr"]

from pathlib import Path

import pandas as pd
from Bio import SeqIO


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
        """Return a pd.DataFrame of sequences with MultiIndex (path, chain_id).

        Returns
        --------
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
        df = pd.DataFrame(sequences, index=index, columns=["sequence"])

        if self.ignore_duplicates:
            df = df.loc[~df["sequence"].duplicated(keep="first")]

        if self.columns is not None:
            df.columns = self.columns

        return df

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
