__author__ = "satvshr"

import pandas as pd
import pytest

from pyaptamer.datasets import (
    load_csv_dataset,
    load_hf_dataset,
)


def test_load_csv():
    """Test loading a CSV dataset."""
    result = load_csv_dataset("dummy_data")
    assert isinstance(result, pd.DataFrame)


def test_load_csv_dataset_file_not_found():
    """Test FileNotFoundError raised when the file does not exist."""
    with pytest.raises(FileNotFoundError, match="Dataset dummy_nonexistent not found"):
        load_csv_dataset("dummy_nonexistent")


def test_hf_dataset_loader_already_downloaded():
    """Test loading a hugging face file when alwaready downloaded locally."""
    result = load_hf_dataset("dummy_data")
    assert isinstance(result, pd.DataFrame)


def test_hf_dataset_loader_file_not_found():
    """Test FileNotFoundError raised when the file does not exist on hugging face."""
    with pytest.raises(FileNotFoundError, match="Dataset dummy_nonexistent not found"):
        load_hf_dataset("dummy_nonexistent")
