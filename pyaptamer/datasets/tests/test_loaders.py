import pytest
from Bio.PDB.Structure import Structure

from pyaptamer.datasets.loader import (
    load_3eiy_structure,
    load_pfoa_structure,
)

LOADERS = [
    load_pfoa_structure,
    load_3eiy_structure,
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
