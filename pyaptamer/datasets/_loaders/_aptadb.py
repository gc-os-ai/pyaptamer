__author__ = ["tarun-227"]
__all__ = ["load_aptadb"]

import os

import pandas as pd

VALID_TABLES = ("interaction", "aptamer", "protein", "molecule", "cell", "other")


def load_aptadb(table=None, select_columns=None):
    """
    Load AptaDB dataset tables as pandas DataFrames.

    AptaDB is a comprehensive aptamer database containing aptamer sequences,
    target information (proteins, molecules, cells), and interaction data.

    Source: https://lmmd.ecust.edu.cn/aptadb

    Parameters
    ----------
    table : str or None, optional
        Which table to load. Must be one of:
        "interaction", "aptamer", "protein", "molecule", "cell", "other".

        If None (default), all six tables are returned as a dict.

    select_columns : list[str] or None, optional
        Column names to keep. If None, returns all columns.
        Only applicable when ``table`` is a string (single table).

    Returns
    -------
    pandas.DataFrame or dict[str, pandas.DataFrame]
        - If ``table`` is a string: a single DataFrame for that table,
          optionally filtered to ``select_columns``.
        - If ``table`` is None: a dict mapping table name to DataFrame
          for all six tables.

    Raises
    ------
    ValueError
        If ``table`` is not one of the valid table names or None.

    Examples
    --------
    >>> from pyaptamer.datasets import load_aptadb
    >>> interaction = load_aptadb("interaction")
    >>> interaction.shape[0] > 0
    True

    >>> all_tables = load_aptadb()
    >>> sorted(all_tables.keys())
    ['aptamer', 'cell', 'interaction', 'molecule', 'other', 'protein']
    """
    if table is not None and table not in VALID_TABLES:
        raise ValueError(f"table must be one of {VALID_TABLES} or None, got {table!r}")

    base_path = os.path.join(os.path.dirname(__file__), "..", "data")

    if table is not None:
        df = _load_single_table(base_path, table)
        if select_columns is not None:
            df = df[select_columns]
        return df

    return {name: _load_single_table(base_path, name) for name in VALID_TABLES}


_OTHER_COLUMNS = ["target_id", "target_name"]


def _load_single_table(base_path, table):
    """Load a single AptaDB CSV table.

    Parameters
    ----------
    base_path : str
        Path to the data directory.
    table : str
        Table name (e.g. "interaction", "aptamer").

    Returns
    -------
    pandas.DataFrame
        The loaded table.
    """
    path = os.path.join(base_path, f"aptadb_{table}.csv")

    kwargs = {"encoding": "latin1"}
    if table == "other":
        kwargs["header"] = None
        kwargs["names"] = _OTHER_COLUMNS

    return pd.read_csv(path, **kwargs)
