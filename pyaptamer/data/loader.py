"""Molecule data loading module."""

from pathlib import Path

import pandas as pd


class MoleculeLoader:
    """Molecule data connector.

    Parameters
    ------------
    path : str, Path, or list thereof
        file location or list of file locations with molecule files
        str can be any of the following... [fill in]

        the following formats are currently supported:

        * `pdb` format
    """

    def __init__(self, path):
        if isinstance(path, str):
            path = [Path(path)]
            self._path = path
        elif isinstance(path, Path):
            self._path = [path]
        elif isinstance(path, list):
            self._path = [Path(p) if isinstance(p, str) else p for p in path]

    def to_df(self):
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
        return pd.DataFrame(seq_list, columns=["sequence"])

    def _determine_type(self, path):
        suffix = path.suffix.lower()
        if suffix == ".pdb":
            return "pdb"
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _load_dispatch(self, path):
        ptype = self._determine_type(path)
        loader = getattr(self, f"_load_{ptype}")
        return loader(path)

    def _load_pdb(self, path):
        """Load a PDB file and extract the primary sequence.

        Parameters
        -----------
        path : Path
            path to PDB file

        Returns
        --------
        str
            primary sequence extracted from PDB file
        """
        sequence = []
        with open(path) as f:
            for line in f:
                if line.startswith("SEQRES"):
                    parts = line.split()
                    seq_parts = parts[4:]  # Skip the first four columns
                    sequence.extend(seq_parts)
        return "".join(sequence)
