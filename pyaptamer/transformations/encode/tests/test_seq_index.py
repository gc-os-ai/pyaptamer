"""Tests for the SequenceIndexEncoder transform."""

__author__ = ["siddharth7113"]

import numpy as np
import pandas as pd
import pytest

from pyaptamer.data import MoleculeLoader
from pyaptamer.transformations.encode import SequenceIndexEncoder


def test_default_vocab_by_hand():
    """ATGCU maps to 0 1 2 3 1 and lower case is accepted."""
    Xt = SequenceIndexEncoder().fit_transform(pd.DataFrame({"seq": ["ATGCU", "atgcu"]}))
    np.testing.assert_array_equal(Xt.to_numpy(), [[0, 1, 2, 3, 1]] * 2)
    assert Xt.to_numpy().dtype == np.int64


def test_custom_vocab():
    """A user vocab replaces the default mapping."""
    enc = SequenceIndexEncoder(vocab={"A": 0, "C": 1, "G": 2, "T": 3})
    Xt = enc.fit_transform(pd.DataFrame({"seq": ["ACGT"]}))
    assert Xt.iloc[0].tolist() == [0, 1, 2, 3]


def test_accepts_moleculeloader_and_keeps_index():
    """A MoleculeLoader input encodes to a frame indexed like the input rows."""
    X = MoleculeLoader(data={"seq": ["ATGC", "GCTA"]})
    Xt = SequenceIndexEncoder().fit_transform(X)
    assert Xt.shape == (2, 4)
    assert Xt.index.tolist() == X.to_dataframe().index.tolist()


def test_rejects_variable_lengths():
    """Sequences of unequal length raise a ValueError."""
    with pytest.raises(ValueError, match="same fixed length"):
        SequenceIndexEncoder().fit_transform(pd.DataFrame({"seq": ["ACGT", "ACG"]}))


def test_rejects_unknown_characters():
    """A character outside the vocab raises a ValueError naming it."""
    with pytest.raises(ValueError, match=r"\['N'\]"):
        SequenceIndexEncoder().fit_transform(pd.DataFrame({"seq": ["ACGN"]}))


def test_rejects_missing_values():
    """A None cell raises a ValueError."""
    with pytest.raises(ValueError, match="None/NaN"):
        SequenceIndexEncoder().fit_transform(pd.DataFrame({"seq": ["ACGT", None]}))


def test_drop_skips_bad_rows_and_keeps_index():
    """handle_unknown='drop' skips None and unknown rows and keeps the others' index."""
    X = pd.DataFrame({"seq": ["ACGT", None, "ACGN", "TTTT"]}, index=[10, 11, 12, 13])
    Xt = SequenceIndexEncoder(handle_unknown="drop").fit_transform(X)
    assert Xt.index.tolist() == [10, 13]
    assert Xt.iloc[1].tolist() == [1, 1, 1, 1]


def test_rejects_invalid_handle_unknown():
    """An unsupported handle_unknown value raises a ValueError."""
    with pytest.raises(ValueError, match="handle_unknown must be one of"):
        SequenceIndexEncoder(handle_unknown="xyz").fit_transform(
            pd.DataFrame({"seq": ["ACGT"]})
        )


def test_rejects_non_string_column():
    """A numeric column raises a TypeError."""
    with pytest.raises(TypeError, match="one str sequence per row"):
        SequenceIndexEncoder().fit_transform(pd.DataFrame({"seq": [1, 2]}))
