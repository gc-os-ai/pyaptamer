__author__ = "rpgv"

import pytest
from datasets import Dataset
from pandas import DataFrame

from pyaptamer.datasets import load_aptacom


@pytest.mark.parametrize("as_df", [True, False])
@pytest.mark.parametrize(
    "filter_entries", ["protein_target", "small_target", "dna_apt", "rna_apt"]
)
def test_download_aptacom(as_df, filter_entries):
    """
    The test_download_aptacom function
    """
    dataset = load_aptacom(as_df, filter_entries)
    if not isinstance(dataset, Dataset | DataFrame):
        raise ValueError(f"Dataset format {type(dataset)} is not DataFrame or Dataset")
