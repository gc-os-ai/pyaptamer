__author__ = ["nennomp", "satvshr"]

import pandas as pd
import pytest
from Bio.PDB.Structure import Structure

from pyaptamer.datasets import (
    load_csv_dataset,
    load_hf_dataset,
    structure_loader,
)

# Replace old specific loaders with generic calls
LOADERS = [
    lambda: structure_loader("5nu7"),
    lambda: structure_loader("1gnh"),
]


@pytest.mark.parametrize("loader", LOADERS)
def test_structure_loader_returns_structure(loader):
    """
    Each generic structure loader should run without error
    and return a Biopython Structure.
    """
    struct = loader()
    assert isinstance(struct, Structure), (
        f"{loader} did not return a Bio.PDB.Structure.Structure"
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
    """Test loading a Hugging Face file when already downloaded locally."""
    result = load_hf_dataset("dummy_data")
    assert isinstance(result, pd.DataFrame)


def test_hf_dataset_loader_file_not_found():
    """Test FileNotFoundError raised when file does not exist on HF."""
    with pytest.raises(FileNotFoundError, match="Dataset dummy_nonexistent not found"):
        load_hf_dataset("dummy_nonexistent")
