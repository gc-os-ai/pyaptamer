__author__ = "rpgv"

import pytest
from pandas import DataFrame

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import load_aptacom, load_aptacom_full


@pytest.mark.parametrize(
    "select_columns",
    [
        ["reference"],
        ["aptamer_chemistry"],
        ["aptamer_name"],
        ["target_name"],
        ["aptamer_sequence"],
        ["origin"],
        ["target_chemistry"],
        ["external_id"],
        ["target_sequence"],
        ["new_affinity"],
    ],
)
def test_load_aptacom_full(select_columns):
    """
    The test_download_aptacom function
    """
    dataset = load_aptacom_full(select_columns)
    if not isinstance(dataset, DataFrame):
        raise ValueError(f"""Dataset format {type(dataset)} 
                is not DataFrame""")


@pytest.mark.parametrize("return_X_y", [True, False])
def test_load_aptacom(return_X_y):
    """x_y returns a MoleculeLoader for molecule data; y stays a DataFrame."""
    result = load_aptacom(return_X_y)
    if return_X_y:
        X, y = result
        assert isinstance(X, MoleculeLoader)
        assert isinstance(y, DataFrame)
    else:
        assert isinstance(result, MoleculeLoader)
