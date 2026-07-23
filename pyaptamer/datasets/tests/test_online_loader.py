__author__ = "satvshr"

import pytest

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import load_from_rcsb


@pytest.mark.parametrize("pdb_id", ["1GNH"])
def test_download_structure(pdb_id):
    """load_from_rcsb downloads a structure and returns a usable MoleculeLoader."""
    loader = load_from_rcsb(pdb_id)
    assert isinstance(loader, MoleculeLoader)

    df = loader.to_dataframe()
    assert not df.empty
