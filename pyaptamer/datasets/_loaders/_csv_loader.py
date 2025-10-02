__author__ = ["satvshr"]
__all__ = ["load_csv_dataset"]

import os

import pandas as pd


def load_csv_dataset(name, target_col, return_X_y=False):
    """
    Load a dataset from a CSV file in DataFrame format.

    Parameters
    ----------
    name : str
        Name of the dataset (file basename without `.csv`) located in the
        package `dataset/data/` directory.
    target_col : str
        Column name in the CSV to use as the target variable.
    return_X_y : bool, optional, default=False
        If True, return (X_df, y_df) as pandas DataFrames.
        If False, return the full DataFrame (features + target).

    Returns
    -------
    pandas.DataFrame or tuple of pandas.DataFrame
        If `return_X_y` is False, returns the full DataFrame with all columns.
        If `return_X_y` is True, returns:
            - X_df : pd.DataFrame of shape (n_samples, n_features)
            - y_df : pd.DataFrame of shape (n_samples, 1)
    """
    path = os.path.join(os.path.dirname(__file__), "..", "data", f"{name}.csv")

    df = pd.read_csv(path)

    if return_X_y:
        X_df = df.drop(columns=[target_col])
        y_df = df[[target_col]]
        return X_df, y_df

    return df
