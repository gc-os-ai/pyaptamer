"""Tests for ApplyToCols."""

__author__ = ["siddharth7113"]

import pandas as pd
import pytest

from pyaptamer.trafos.compose import ApplyToCols
from pyaptamer.trafos.encode import KMerFrequencies
from pyaptamer.trafos.transform import PrimerTrimmer


def _frame():
    return pd.DataFrame(
        {
            "id": ["r1", "r2", "r3"],
            "sequence": ["AAACGTTT", "AAAGGTTT", "GGGG"],
            "round": [1, 1, 2],
        },
        index=["a", "b", "c"],
    )


def test_named_column_is_transformed_and_others_kept():
    """Only the named column is encoded; the other string column passes through."""
    Xt = ApplyToCols(KMerFrequencies(k=1), cols="sequence").fit_transform(_frame())
    assert list(Xt.columns[:1]) == ["id"]
    assert list(Xt.columns[-1:]) == ["round"]
    assert [c for c in Xt.columns if c.startswith("sequence__")] == [
        "sequence__0",
        "sequence__1",
        "sequence__2",
        "sequence__3",
    ]
    assert Xt["id"].tolist() == ["r1", "r2", "r3"]


def test_dropped_rows_leave_every_column():
    """A trimmer that drops a read drops that row from the other columns too."""
    trimmer = PrimerTrimmer(start_primer="AAA", end_primer="TTT", variable_length=2)
    Xt = ApplyToCols(trimmer, cols="sequence").fit_transform(_frame())
    assert Xt.index.tolist() == ["a", "b"]
    assert Xt["sequence"].tolist() == ["CG", "GG"]
    assert Xt["id"].tolist() == ["r1", "r2"]


def test_inner_transformer_is_cloned():
    """fit works on a clone; the transformer passed in stays unfitted."""
    inner = KMerFrequencies(k=1)
    wrapper = ApplyToCols(inner, cols="sequence").fit(_frame())
    assert inner.is_fitted is False
    assert wrapper.transformer_.is_fitted is True


def test_missing_column_raises():
    """A column name that X does not have is an error."""
    with pytest.raises(ValueError, match="not a column"):
        ApplyToCols(KMerFrequencies(k=1), cols="reads").fit(_frame())


def test_cols_must_be_one_name():
    """cols must be a single column name, not a list."""
    with pytest.raises(TypeError, match="one column name"):
        ApplyToCols(KMerFrequencies(k=1), cols=["sequence"]).fit(_frame())


def test_transformed_block_keeps_column_position():
    """The encoded columns sit where the source column was."""
    X = pd.DataFrame({"round": [1], "sequence": ["ACGT"], "gc": [0.5]})
    Xt = ApplyToCols(KMerFrequencies(k=1), cols="sequence").fit_transform(X)
    assert Xt.columns.tolist() == [
        "round",
        "sequence__0",
        "sequence__1",
        "sequence__2",
        "sequence__3",
        "gc",
    ]


def test_dropped_rows_with_duplicate_index_raises():
    """Dropping rows needs a unique row index to realign the other columns."""
    X = pd.DataFrame({"sequence": ["AAACGTTT", "GGGG"], "round": [1, 2]}, index=[0, 0])
    trimmer = PrimerTrimmer(start_primer="AAA", end_primer="TTT", variable_length=2)
    with pytest.raises(ValueError, match="unique"):
        ApplyToCols(trimmer, cols="sequence").fit_transform(X)


def test_duplicate_index_without_dropped_rows_is_fine():
    """A repeated row index is fine when the transformer keeps every row."""
    X = pd.DataFrame({"sequence": ["ACGT", "GGCC"], "round": [1, 2]}, index=[0, 0])
    Xt = ApplyToCols(KMerFrequencies(k=1), cols="sequence").fit_transform(X)
    assert Xt["round"].tolist() == [1, 2]


def test_output_name_clash_raises():
    """A column already named like an output column is an error, not a duplicate."""
    X = pd.DataFrame({"sequence": ["ACGT"], "sequence__0": [1.0]})
    with pytest.raises(ValueError, match="sequence__0"):
        ApplyToCols(KMerFrequencies(k=1), cols="sequence").fit_transform(X)


def test_named_column_reaches_the_transformer_check():
    """A bag-tiled column is handed to the transformer, whose own check rejects it."""
    X = pd.DataFrame({"target": ["thrombin"], "selex": [["AAACGTTT", "GGGG"]]})
    trimmer = PrimerTrimmer(start_primer="AAA", end_primer="TTT", variable_length=2)
    with pytest.raises(TypeError, match="tiling"):
        ApplyToCols(trimmer, cols="selex").fit_transform(X)
