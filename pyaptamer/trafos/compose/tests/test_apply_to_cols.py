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
