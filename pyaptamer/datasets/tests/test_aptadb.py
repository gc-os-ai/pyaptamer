__author__ = ["tarun-227"]

import pandas as pd
import pytest

from pyaptamer.datasets import load_aptadb

VALID_TABLES = ("interaction", "aptamer", "protein", "molecule", "cell", "other")

EXPECTED_COLUMNS = {
    "interaction": ["Index", "Apta_index", "TargetID", "Sequence"],
    "aptamer": ["Apta_index", "Sequence", "Aptamer Chemistry"],
    "protein": ["Uni-port ID", "Entry name", "Protein names"],
    "molecule": ["Pubchem ID", "Titles", "Molecular Formula"],
    "cell": ["ATCC", "Name"],
    "other": ["target_id", "target_name"],
}


@pytest.mark.parametrize("table", VALID_TABLES)
def test_load_aptadb_single_table(table):
    """Test that each table loads as a non-empty DataFrame."""
    df = load_aptadb(table)
    assert isinstance(df, pd.DataFrame), f"{table} should return a DataFrame"
    assert df.shape[0] > 0, f"{table} should not be empty"
    assert df.shape[1] > 0, f"{table} should have columns"


@pytest.mark.parametrize("table", VALID_TABLES)
def test_load_aptadb_expected_columns(table):
    """Test that each table contains expected columns."""
    df = load_aptadb(table)
    for col in EXPECTED_COLUMNS[table]:
        assert col in df.columns, f"Column {col!r} missing from {table} table"


def test_load_aptadb_all_tables():
    """Test that loading with table=None returns a dict of all tables."""
    result = load_aptadb()
    assert isinstance(result, dict), "Should return a dict when table=None"
    assert set(result.keys()) == set(VALID_TABLES), (
        "Dict keys should match valid table names"
    )
    for name, df in result.items():
        assert isinstance(df, pd.DataFrame), f"{name} should be a DataFrame"
        assert df.shape[0] > 0, f"{name} should not be empty"


def test_load_aptadb_select_columns():
    """Test column selection on a single table."""
    cols = ["Apta_index", "Sequence"]
    df = load_aptadb("interaction", select_columns=cols)
    assert list(df.columns) == cols, "Should return only selected columns"


def test_load_aptadb_invalid_table():
    """Test that an invalid table name raises ValueError."""
    with pytest.raises(ValueError, match="table must be one of"):
        load_aptadb("nonexistent")
