__author__ = ["nennomp", "satvshr"]

import pandas as pd
import pytest
from Bio.PDB.Structure import Structure

from pyaptamer.datasets import (
    load_1gnh_structure,
    load_csv_dataset,
    load_hf_dataset,
    load_pfoa_structure,
)

LOADERS = [
    load_pfoa_structure,
    load_1gnh_structure,
]


@pytest.mark.parametrize("loader", LOADERS)
def test_loader_returns_structure(loader):
    """
    Each loader should run without error and return a Biopython Structure.
    """
    struct = loader()
    assert isinstance(struct, Structure), (
        f"{loader.__name__}() did not return a Bio.PDB.Structure.Structure"
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
