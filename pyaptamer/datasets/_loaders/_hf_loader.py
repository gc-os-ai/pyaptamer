__author__ = ["nennomp"]
__all__ = ["load_hf_dataset"]

import os

import pandas as pd
from datasets import load_dataset

from pyaptamer.datasets._loaders._csv_loader import load_csv_dataset


def load_hf_dataset(name: str, store: bool = True) -> pd.DataFrame:
    """Load a dataset from Hugging Face (HF).

    The method downloads the specified dataset and saves it as a CSV file. If it
    already exists locally, it loads the dataset from the local file.

    Parameters
    ----------
    name : str
        Name of the dataset to load from Hugging Face.
    store : bool, optional, default=True
        If True, the dataset will be locally saved as a CSV file. If False, the dataset
        will be download but only kept in-memory and not saved to disk.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the downloaded/loaded dataset.
    """
    path = os.path.relpath(
        os.path.join(os.path.dirname(__file__), "..", "data", f"{name}.csv")
    )

    # use local file if it exists
    if os.path.exists(path):
        print(f"Dataset {name} already exists locally at {path}.")
        return load_csv_dataset(name)

    print(f"Downloading {name}...")
    dataset = load_dataset(f"gcos/pyaptamer-{name}")
    dataset = dataset["train"].to_pandas()

    if store:
        print(f"Saving to {path}")
        dataset.to_csv(path, index=False)

    return dataset
