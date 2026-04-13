__author__ = "satvshr"

import pytest

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import load_from_rcsb


@pytest.mark.parametrize("pdb_id", ["1GNH"])
def test_download_structure(pdb_id):
    """
    The download_structure function works correctly
    for valid PDB IDs.
    """
    loader = load_from_rcsb(pdb_id)
    assert isinstance(loader, MoleculeLoader), (
        f"Expected a MoleculeLoader, got {type(loader)}"
    )

    df_seq = loader.to_df_seq()
    assert not df_seq.empty, "Expected to_df_seq() to return a non-empty DataFrame"
