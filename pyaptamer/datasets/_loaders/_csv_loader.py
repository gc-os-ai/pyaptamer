__author__ = ["nennomp"]
__all__ = ["load_csv_dataset"]

import os

import pandas as pd


def load_csv_dataset(name: str) -> pd.DataFrame:
    """Load a dataset from a CSV file.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the dataset loaded from the CSV file.

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist.
    """
    path = os.path.relpath(
        os.path.join(os.path.dirname(__file__), "..", "data", f"{name}.csv")
    )

    if os.path.exists(path):
        dataset = pd.read_csv(path)
        return dataset
    else:
        raise FileNotFoundError(
            f"Dataset {name} not found at {path}. Please ensure the file exists."
        )
