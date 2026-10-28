"""Input check shared by the end-to-end classes and the transformer base."""

__author__ = ["siddharth7113"]
__all__ = ["check_X"]

import pandas as pd

from pyaptamer.data.loader import MoleculeLoader


def check_X(X, cols=None, owner=None):  # noqa: N802
    """Return ``X`` as a DataFrame, reduced to the named columns.

    Parameters
    ----------
    X : MoleculeLoader or pd.DataFrame
        Input data. A MoleculeLoader is converted with ``to_dataframe``.
    cols : dict of str to str or None, default=None
        Column parameters of the caller, parameter name to column name, as in
        ``{"aptamer_col": "aptamer"}``. If None, all columns are kept.
    owner : str or None, default=None
        Name of the class whose parameters ``cols`` are, used in the error
        message for a missing column.

    Returns
    -------
    pd.DataFrame
        ``X`` as a DataFrame with the columns in ``cols``, in that order, or
        all columns if ``cols`` is None.

    Raises
    ------
    TypeError
        If ``X`` is neither a MoleculeLoader nor a DataFrame.
    ValueError
        If a column named in ``cols`` is not in ``X``.
    """
    if isinstance(X, MoleculeLoader):
        X = X.to_dataframe()
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a MoleculeLoader instance or a pandas DataFrame. "
            f"Got {type(X)} instead."
        )
    if cols is None:
        return X
    missing = [c for c in cols.values() if c not in X.columns]
    if missing:
        params = ", ".join(f"{p}=..." for p in cols)
        raise ValueError(
            f"X has no column {missing}. X has columns {list(X.columns)}. "
            f"Pass the column names as {owner}({params})."
        )
    return X[list(cols.values())]
