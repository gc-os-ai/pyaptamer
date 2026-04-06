__author__ = "Jayant-kernel"

import pytest
from pandas import DataFrame

from pyaptamer.datasets import (
    load_aptadb_aptamer,
    load_aptadb_cell,
    load_aptadb_interaction,
    load_aptadb_molecule,
    load_aptadb_other,
    load_aptadb_protein,
)


@pytest.mark.parametrize(
    "loader",
    [
        load_aptadb_interaction,
        load_aptadb_aptamer,
        load_aptadb_protein,
        load_aptadb_molecule,
        load_aptadb_cell,
        load_aptadb_other,
    ],
)
def test_load_aptadb_returns_dataframe(loader):
    """Each AptaDB loader should return a non-empty pandas DataFrame."""
    df = loader()
    assert isinstance(df, DataFrame), (
        f"{loader.__name__} returned {type(df)}, expected DataFrame"
    )
    assert len(df) > 0, f"{loader.__name__} returned an empty DataFrame"


def test_load_aptadb_interaction_has_columns():
    """Interaction table should have at least one column."""
    df = load_aptadb_interaction()
    assert isinstance(df, DataFrame)
    assert df.shape[1] > 0


def test_load_aptadb_cache(tmp_path):
    """Loader should respect a custom cache_dir and not re-download."""
    df1 = load_aptadb_interaction(cache_dir=tmp_path)
    assert isinstance(df1, DataFrame)

    # Second call must use cache (no network needed)
    df2 = load_aptadb_interaction(cache_dir=tmp_path)
    assert df1.shape == df2.shape


def test_load_aptadb_force_download(tmp_path):
    """force_download=True should re-download and still return a DataFrame."""
    df = load_aptadb_interaction(cache_dir=tmp_path, force_download=True)
    assert isinstance(df, DataFrame)
    assert len(df) > 0
