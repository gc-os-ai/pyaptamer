__author__ = "rpgv"

import pytest
from datasets import Dataset
from pandas import DataFrame

from pyaptamer.datasets import load_aptacom


@pytest.mark.parametrize("as_df", [True, False])
def test_download_aptacom(as_df):
    """
    The test_download_aptacom function
    """
    dataset = load_aptacom(as_df)
    if not isinstance(dataset, Dataset | DataFrame):
        raise ValueError(f"Dataset format {type(dataset)} is not DataFrame or Dataset")
