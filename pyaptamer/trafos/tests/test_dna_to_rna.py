"""Tests for the DNAToRNA transformer."""

__author__ = ["siddharth7113"]

import pandas as pd

from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.transform import DNAToRNA


def test_t_becomes_u():
    """Every T is replaced by U and letters are upper-cased."""
    X = pd.DataFrame({"seq": ["ACGT", "ttgg"]})
    Xt = DNAToRNA().fit_transform(X)
    assert Xt["seq"].tolist() == ["ACGU", "UUGG"]


def test_rna_is_unchanged():
    """A sequence without T comes back as it was."""
    X = pd.DataFrame({"seq": ["ACGU"]})
    Xt = DNAToRNA().fit_transform(X)
    assert Xt["seq"].tolist() == ["ACGU"]


def test_column_name_and_index_are_kept():
    """The output keeps the input column name and row index."""
    X = pd.DataFrame({"aptamer": ["TTT", "AAA"]}, index=["x", "y"])
    Xt = DNAToRNA().fit_transform(X)
    assert Xt.columns.tolist() == ["aptamer"]
    assert Xt.index.tolist() == ["x", "y"]


def test_moleculeloader_input():
    """A MoleculeLoader is accepted like a one-column frame."""
    X = MoleculeLoader(data={"seq": ["ACGT"]})
    assert DNAToRNA().fit_transform(X)["seq"].tolist() == ["ACGU"]


def test_missing_cell_passes_through():
    """A missing cell stays missing instead of raising."""
    X = pd.DataFrame({"seq": ["ACGT", None]})
    Xt = DNAToRNA().fit_transform(X)
    assert Xt["seq"].iloc[0] == "ACGU"
    assert pd.isna(Xt["seq"].iloc[1])
