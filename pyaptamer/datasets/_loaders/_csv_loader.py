__author__ = ["nennomp"]
__all__ = ["load_csv_dataset"]

import os

import pandas as pd


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
    FileNotFoundError
        If the specified CSV file does not exist.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "data", f"{name}.csv")

    if os.path.exists(path):
        return pd.read_csv(path, keep_default_na=keep_default_na, na_values=na_values)
    else:
        raise FileNotFoundError(
            f"Dataset {name} not found at {path}. Please ensure the file exists."
        )
