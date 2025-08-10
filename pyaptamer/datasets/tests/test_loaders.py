import pandas as pd
import pytest
from Bio.PDB.Structure import Structure

from pyaptamer.datasets._loaders import (
    load_1gnh_structure,
    load_li_dataset,
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


def test_loader_li_dataset():
    """Test that the (csv) Li dataset loader works correctly."""
    train, test, freqs = load_li_dataset()
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(freqs, pd.DataFrame)
    assert len(train) == 2320
    assert len(test) == 580
    assert len(freqs) == 8420
