__author__ = ["nennomp"]
__all__ = ["load_csv_dataset"]

import os

import pandas as pd

def _validate_dataset_name(name: str) -> None:
    """Reject names that could cause path traversal."""
    if not name or name != name.strip():
        raise ValueError("Dataset name must be non-empty and not start/end with whitespace")
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Dataset name must not contain path separators or '..'.")


def load_csv_dataset(
    name: str, keep_default_na: bool = True, na_values: list[str] | None = None
) -> pd.DataFrame:
    """Load a dataset from a CSV file.

    Parameters
    ----------
    name : str
        Name of the dataset to load.
    keep_default_na : bool, optional, default=True
        Whether to keep the default NaN values or not. Depending on `na_values`.
    na_values : list[str] | None, optional, default=None
        Additional strings to recognize as NaN values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the dataset loaded from the CSV file.

    Raises
    ------
    ValueError
        If the dataset name is invalid (empty, contains path separators, or '..').
    FileNotFoundError
        If the specified CSV file does not exist.
    """
    _validate_dataset_name(name)
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", f"{name}.csv")
    )

    if os.path.exists(path):
        return pd.read_csv(path, keep_default_na=keep_default_na, na_values=na_values)
    else:
        raise FileNotFoundError(
            f"Dataset {name} not found at {path}. Please ensure the file exists."
        )
