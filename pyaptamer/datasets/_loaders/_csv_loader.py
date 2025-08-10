__author__ = ["nennomp"]
__all__ = ["load_csv_dataset", "load_li_dataset"]

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
    path = os.path.join(os.path.dirname(__file__), "..", "data", f"{name}.csv")

    if os.path.exists(path):
        dataset = pd.read_csv(path)
        return dataset
    else:
        raise FileNotFoundError(
            f"Dataset {name} not found at {path}. Please ensure the file exists."
        )

def load_li_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the aptamer-protein interaction benchmark from [1]_.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Dataframes containing the train split, test split, and protein triplet 
        frequencies.

    References
    ----------
    .. [1] Li, Bi-Qing, et al. "Prediction of aptamer-target interacting pairs with 
    pseudo-amino acid composition." PLoS One 9.1 (2014): e86729.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "data")

    # train split
    train = load_csv_dataset(os.path.join(path, "train_li2014"))
     # test split
    test = load_csv_dataset(os.path.join(path, "test_li2014"))

    # protein triplets frequencies
    freqs = load_csv_dataset(os.path.join(path, "protein_word_freq"))

    return (train, test, freqs)