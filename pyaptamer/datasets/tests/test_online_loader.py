__author__ = "satvshr"

import os

import pytest

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import load_1gnh, load_from_rcsb


@pytest.mark.parametrize("pdb_id", ["1GNH"])
def test_download_structure(pdb_id, tmp_path):
    """load_from_rcsb stores <id>.pdb in pdir and matches the bundled loader."""
    loader = load_from_rcsb(pdb_id, pdir=tmp_path)
    assert isinstance(loader, MoleculeLoader)

    assert os.listdir(tmp_path) == ["1gnh.pdb"]

    df = loader.to_dataframe()
    assert not df.empty
    assert df.equals(load_1gnh().to_dataframe())

    df = load_from_rcsb(pdb_id, pdir=tmp_path, tiling="samples").to_dataframe()
    assert df.equals(load_1gnh(tiling="samples").to_dataframe())
