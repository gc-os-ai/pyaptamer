__author__ = "satvshr"

import pandas as pd

from pyaptamer.datasets._loaders._csv_loader import load_csv_dataset

DATASET_NAME = "train_li2014"
TARGET_COL = "label"


def test_load_csv_return_x_y():
    """
    When return_X_y=True the loader should return two pandas objects (X_df, y_df).
    """
    X_df, y_df = load_csv_dataset(DATASET_NAME, target_col=TARGET_COL, return_X_y=True)

    assert isinstance(X_df, pd.DataFrame), "X should be a pandas DataFrame"
    assert isinstance(y_df, pd.DataFrame), (
        "y should be a pandas DataFrame (single-column)"
    )
    assert X_df.shape[0] == y_df.shape[0], "Number of samples in X and y must match"
    assert X_df.ndim == 2, "X should be a 2D DataFrame (n_samples, n_features)"
    assert y_df.shape[1] == 1, "y should have a single column"


def test_load_csv_returns_df():
    """
    When return_X_y=False the loader should return the full DataFrame containing the
    target column.
    """
    df = load_csv_dataset(DATASET_NAME, target_col=TARGET_COL, return_X_y=False)

    assert isinstance(df, pd.DataFrame), "Returned object should be a pandas DataFrame"
    assert TARGET_COL in df.columns, (
        f"DataFrame must contain the target column '{TARGET_COL}'"
    )
    assert df.shape[0] > 0, "DataFrame should not be empty"
