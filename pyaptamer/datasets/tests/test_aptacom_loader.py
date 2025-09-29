__author__ = "rpgv"

import pytest
from datasets import Dataset
from pandas import DataFrame

from pyaptamer.datasets import load_aptacom


@pytest.mark.parametrize("as_df", [True, False])
def test_download_aptacom(as_df):
    """
    The download_aptacom function 
    """
    dataset = load_aptacom(as_df)
    if not isinstance(dataset, (Dataset, DataFrame)):
        raise ValueError(f"Expected a datasets.Dataset or pandas.DataFrame but got  {type(dataset)}")

