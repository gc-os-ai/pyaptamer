__author__ = "rpgv"

import pytest
from pandas import DataFrame

from pyaptamer.datasets import load_aptacom_full, load_aptacom_xy


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
def test_download_aptacom_x_y(return_X_y):
    """
    The test_download_aptacom function
    """
    dataset = load_aptacom_xy(return_X_y)
    if not isinstance(dataset, tuple | DataFrame):
        raise ValueError(f"""Dataset format {type(dataset)} 
            is not X, y tuple or DataFrame""")
