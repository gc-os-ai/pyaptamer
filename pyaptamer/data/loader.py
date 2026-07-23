"""Molecule data loading module."""

__all__ = ["MoleculeLoader"]
__author__ = ["fkiraly", "satvshr", "siddharth7113"]

from itertools import product
from pathlib import Path

import pandas as pd
from Bio import SeqIO


class MoleculeLoader:
    """Lazy 2D container for molecule data and binding tables.

    Holds tabular data whose cells may be file references (FASTA, PDB,
    FASTQ, ...) or in-memory values (sequence strings, numbers). Files are
    not parsed until ``to_dataframe`` is called.

    Parameters
    ----------
    data : dict, list of lists, or pandas.DataFrame
        2D data coercible by ``pandas.DataFrame``. Keys (or columns) become
        column names; cells hold file paths, sequence strings, or primitives.
    tiling : str, default="bag"
        How multi-sequence files are materialized. One of ``"bag"``,
        ``"features"``, ``"samples"``, ``"samples_product"``, ``"first"``,
        ``"concat"``.
    indexing : str, default="preserve"
        What happens to file-internal IDs (PDB chain IDs, FASTA headers).
        One of ``"new"``, ``"preserve"``, ``"keep_as_column"``.
    multiindex : str, default="flatten"
        Row-index shape after sequences expand into rows. One of
        ``"flatten"``, ``"multiindex"``, ``"auto"``.
    ignore_duplicates : bool, default=False
        If True, deduplicate identical sequences within each file before
        tiling is applied.

    Notes
    -----
    Supported file formats are anything :func:`Bio.SeqIO.parse` can read
    (FASTA, GenBank, EMBL, FASTQ, ...); PDB files are read through the
    ``pdb-seqres`` parser. The format is inferred from the file suffix.

    Examples
    --------
    Build a table from in-memory paired sequences and binding values. With
    sequences already in memory, ``to_dataframe`` is a no-op:

    >>> from pyaptamer.data.loader import MoleculeLoader
    >>> loader = MoleculeLoader(
    ...     data={"target": ["ACGT", "TTGG"], "binding": [0.4, 0.6]}
    ... )
    >>> df = loader.to_dataframe()
    >>> list(df.columns)
    ['target', 'binding']
    >>> df["target"].tolist()
    ['ACGT', 'TTGG']
    """

    def __init__(
        self,
        data,
        tiling="bag",
        indexing="preserve",
        multiindex="flatten",
        ignore_duplicates=False,
    ):
        valid_tilings = {
            "bag",
            "concat",
            "first",
            "features",
            "samples",
            "samples_product",
        }
        if tiling not in valid_tilings:
            raise ValueError(
                f"tiling must be one of {sorted(valid_tilings)}, got {tiling!r}"
            )

        valid_indexings = {"new", "preserve", "keep_as_column"}
        if indexing not in valid_indexings:
            raise ValueError(
                f"indexing must be one of {sorted(valid_indexings)}, got {indexing!r}"
            )

        valid_multiindex = {"flatten", "multiindex", "auto"}
        if multiindex not in valid_multiindex:
            raise ValueError(
                f"multiindex must be one of {sorted(valid_multiindex)}, "
                f"got {multiindex!r}"
            )

        self.data = data
        self.tiling = tiling
        self.indexing = indexing
        self.multiindex = multiindex
        self.ignore_duplicates = ignore_duplicates

    def to_dataframe(self):
        """Materialize the loader into a :class:`pandas.DataFrame`.

        File-path cells are parsed and reshaped according to ``tiling``,
        ``indexing`` and ``multiindex``.

        Returns
        -------
        pandas.DataFrame
            The materialized table. Its exact shape depends on ``tiling``.

        Examples
        --------
        A multi-chain PDB under ``tiling="bag"`` becomes a list-valued cell,
        while a single-chain file stays a plain string:

        >>> loader = MoleculeLoader(data={"target": ["1gnh.pdb"]})
        >>> loader.to_dataframe()["target"].iloc[0]  # doctest: +SKIP
        ['QTDMSRK...', 'QTDMSRK...', ...]

        A SELEX FASTQ explodes to one row per read with ``tiling="samples"``:

        >>> loader = MoleculeLoader(data={"selex": ["round5.fastq"]}, tiling="samples")
        >>> loader.to_dataframe()  # doctest: +SKIP

        For comprehensive, file-based walkthroughs across formats (PDB, FASTA,
        FASTQ) and tilings, see the tutorial notebooks in ``examples/``.
        """
        df = pd.DataFrame(self.data)

        if self.tiling in {"bag", "concat", "first"}:
            return df.map(self._materialize_cell)
        if self.tiling == "features":
            return self._tile_features(df)
        return self._tile_samples(df, product_mode=self.tiling == "samples_product")

    def _cell_records(self, value):
        """Normalize a cell to ``(is_file, [(chain_id, sequence), ...])``.

        Non-path cells return ``(False, [(None, value)])`` so callers can
        treat literals uniformly. Path cells are parsed and, when
        ``ignore_duplicates`` is set, deduplicated by sequence (first
        occurrence wins).

        Parameters
        ----------
        value : object
            A single DataFrame cell.

        Returns
        -------
        tuple of (bool, list of (object, str))
            Whether the cell was a file, and its (chain_id, sequence) records.
        """
        if not self._is_path_like(value):
            return False, [(None, value)]

        loaded = self._load_dispatch(Path(value))
        records = list(
            zip(
                loaded["chain_id"].tolist(),
                loaded["sequence"].tolist(),
                strict=True,
            )
        )

        if self.ignore_duplicates:
            seen = set()
            unique = []
            for chain_id, seq in records:
                if seq in seen:
                    continue
                seen.add(seq)
                unique.append((chain_id, seq))
            records = unique

        return True, records

    def _materialize_cell(self, value):
        """Render one cell for the per-cell tilings ``bag``/``concat``/``first``.

        Non-path values are returned unchanged. For path cells:

        - ``bag``: a plain ``str`` for a single-sequence file, else a
          ``list[str]``.
        - ``concat``: all sequences joined into one string, ordered
          alphabetically by chain id (no separator).
        - ``first``: the first sequence only.

        Parameters
        ----------
        value : object
            A single DataFrame cell.

        Returns
        -------
        object
            The original value, or the rendered sequence(s).
        """
        is_file, records = self._cell_records(value)
        if not is_file:
            return value

        sequences = [seq for _, seq in records]

        if self.tiling == "first":
            return sequences[0]
        if self.tiling == "concat":
            ordered = sorted(records, key=lambda rec: str(rec[0]))
            return "".join(seq for _, seq in ordered)
        # bag
        return sequences[0] if len(sequences) == 1 else sequences

    def _is_path_like(self, value):
        """Return True if ``value`` looks like a file reference.

        A cell is treated as a file when it is a string with a non-empty
        suffix (e.g. ``"1gnh.pdb"``). Sequence strings such as ``"ACGT"`` have
        no suffix and are left untouched.

        Notes
        -----
        Failure mode: a sequence string that happens to contain a dot would be
        misread as a path. Biological sequences do not contain dots, so this
        heuristic is acceptable for the initial implementation.
        """
        return isinstance(value, str) and Path(value).suffix != ""

    def _tile_features(self, df):
        """Expand multi-sequence file cells into multiple columns.

        A column whose cells hold multi-sequence files is spread into
        ``<col>_0``, ``<col>_1``, ... (one column per sequence). Single-
        sequence columns keep their name. Rows with fewer sequences are
        padded with ``None``.
        """
        new_data = {}
        for col in df.columns:
            loaded = [self._cell_records(v) for v in df[col]]
            if not any(is_file for is_file, _ in loaded):
                new_data[col] = list(df[col])
                continue

            max_n = max((len(recs) for is_file, recs in loaded if is_file), default=1)
            if max_n == 1:
                new_data[col] = [
                    recs[0][1] if is_file else value
                    for (is_file, recs), value in zip(loaded, df[col], strict=True)
                ]
                continue

            for i in range(max_n):
                column = []
                for (is_file, recs), value in zip(loaded, df[col], strict=True):
                    if is_file:
                        column.append(recs[i][1] if i < len(recs) else None)
                    else:
                        column.append(value if i == 0 else None)
                new_data[f"{col}_{i}"] = column

        return pd.DataFrame(new_data, index=df.index)

    def _tile_samples(self, df, product_mode):
        """Explode file cells into rows (``samples`` / ``samples_product``).

        Each file cell contributes one row per sequence. Literal columns are
        repeated. With several file columns in a row, ``samples`` zips them
        (and requires matching counts), while ``samples_product`` takes their
        cartesian product.
        """
        rows = []
        parents = []
        seq_ids = []
        chain_id_cols = {col: [] for col in df.columns}
        expanded = False

        for pos, (_, row) in enumerate(df.iterrows()):
            col_recs = {col: self._cell_records(row[col]) for col in df.columns}
            file_cols = [col for col in df.columns if col_recs[col][0]]

            if not file_cols:
                rows.append({col: row[col] for col in df.columns})
                parents.append(pos)
                seq_ids.append(None)
                for col in df.columns:
                    chain_id_cols[col].append(None)
                continue

            recs_per_col = [col_recs[col][1] for col in file_cols]
            if product_mode:
                combos = list(product(*recs_per_col))
            elif len(file_cols) > 1:
                raise ValueError(
                    "tiling='samples' cannot expand more than one file column "
                    f"in the same row (columns {file_cols}). A multi-sequence "
                    "file such as a multi-chain PDB is one molecule, not parallel "
                    "samples, so pairing columns by position would be meaningless. "
                    "Combine such columns with tiling='concat' first, or use "
                    "per-column tiling (planned)."
                )
            else:
                combos = list(zip(*recs_per_col, strict=True))

            if len(combos) > 1:
                expanded = True

            for combo in combos:
                new_row = {}
                cid_map = {}
                for col in df.columns:
                    if col in file_cols:
                        chain_id, seq = combo[file_cols.index(col)]
                        new_row[col] = seq
                        cid_map[col] = chain_id
                    else:
                        new_row[col] = row[col]
                        cid_map[col] = None
                rows.append(new_row)
                parents.append(pos)
                if len(file_cols) == 1:
                    seq_ids.append(cid_map[file_cols[0]])
                else:
                    seq_ids.append(tuple(cid_map[col] for col in file_cols))
                for col in df.columns:
                    chain_id_cols[col].append(cid_map[col])

        out = pd.DataFrame(rows, columns=list(df.columns))
        return self._finalize_index(
            out, df.index, parents, seq_ids, chain_id_cols, expanded
        )

    def _finalize_index(
        self, out, orig_index, parents, seq_ids, chain_id_cols, expanded
    ):
        """Apply ``indexing`` and ``multiindex`` to an exploded frame."""
        if self.indexing == "keep_as_column":
            for col, cids in chain_id_cols.items():
                if any(cid is not None for cid in cids):
                    out[f"{col}_chain_id"] = cids
            out.index = pd.RangeIndex(len(out))
            return out

        if self.indexing == "new":
            out.index = pd.RangeIndex(len(out))
            return out

        parent_labels = [orig_index[p] for p in parents]

        norm_ids = []
        for sid in seq_ids:
            if sid is None:
                norm_ids.append("")
            elif isinstance(sid, tuple):
                norm_ids.append("_".join(str(x) for x in sid))
            else:
                norm_ids.append(str(sid))

        use_multi = self.multiindex == "multiindex" or (
            self.multiindex == "auto" and expanded
        )

        if use_multi:
            out.index = pd.MultiIndex.from_arrays(
                [parent_labels, norm_ids], names=["row", "sequence"]
            )
        else:
            out.index = [
                f"{label}__{sid}" if sid != "" else f"{label}"
                for label, sid in zip(parent_labels, norm_ids, strict=True)
            ]
        return out

    def _determine_type(self, path):
        """Return the format string inferred from the file suffix.

        Parameters
        ----------
        path : Path
            Path to a file.

        Returns
        -------
        str or None
            Format string such as "pdb", "fasta", "genbank"; ``None`` when the
            path has no suffix.
        """
        suffix = path.suffix.lower()
        return suffix.lstrip(".") if suffix else None

    def _load_dispatch(self, path):
        """Dispatch the loader based on file type.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["chain_id", "sequence"]``.
        """
        fmt = self._determine_type(path)

        if fmt is None:
            raise ValueError(f"Could not determine file format for '{path}'.")

        if fmt == "pdb":
            return self._load_pdb_seq(path)

        return self._load_seqio(path, fmt)

    def _load_pdb_seq(self, path):
        """Load a PDB file and extract the amino-acid sequences.

        Parameters
        ----------
        path : Path
            path to PDB file

        Returns
        -------
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
        if not records:
            raise ValueError(f"No sequences found in {path}")

        return pd.DataFrame.from_records(records, columns=["chain_id", "sequence"])

    def _load_seqio(self, path, fmt):
        """Load any non-PDB file format supported by Biopython SeqIO.

        Notes
        -----
        For non-PDB formats there is usually no literal chain concept.
        To preserve the existing abstract datatype, ``record.id`` is stored
        in the ``chain_id`` column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["chain_id", "sequence"]``.
        """
        with open(path) as handle:
            records = list(SeqIO.parse(handle, fmt))

        rows = [
            {
                "chain_id": record.id,
                "sequence": str(record.seq),
            }
            for record in records
        ]

        if not rows:
            raise ValueError(f"No sequences found in {path}")

        return pd.DataFrame.from_records(rows, columns=["chain_id", "sequence"])
