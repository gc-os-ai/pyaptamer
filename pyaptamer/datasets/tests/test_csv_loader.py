__author__ = "satvshr"

import pandas as pd
import pytest

from pyaptamer.datasets._loaders._csv_loader import load_csv_dataset

DATASET_NAME = "train_li2014"
TARGET_COL = "label"


def test_load_csv():
    """Test loading a CSV dataset."""
    result = load_csv_dataset("dummy_data")
    assert isinstance(result, pd.DataFrame)


def test_load_csv_dataset_file_not_found():
    """Test FileNotFoundError raised when the file does not exist."""
    with pytest.raises(FileNotFoundError, match="Dataset dummy_nonexistent not found"):
        load_csv_dataset("dummy_nonexistent")


def test_load_csv_returns_df():
    """
    When return_X_y=False the loader should return the full DataFrame containing the
    target column.
    """
    df = load_csv_dataset(DATASET_NAME)

    assert isinstance(df, pd.DataFrame), "Returned object should be a pandas DataFrame"
    assert TARGET_COL in df.columns, (
        f"DataFrame must contain the target column '{TARGET_COL}'"
    )
    assert df.shape[0] > 0, "DataFrame should not be empty"
