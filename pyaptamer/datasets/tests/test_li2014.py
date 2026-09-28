__author__ = "siddharth7113"

import pandas as pd
import pytest

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets._loaders._li2014 import load_li2014


@pytest.mark.parametrize(
    "split",
    [None, "train", "test"],
)
def test_load_li2014_single_loader(split):
    """Default returns one MoleculeLoader over aptamer, protein and label."""
    loader = load_li2014(split=split)

    assert isinstance(loader, MoleculeLoader)

    df = loader.to_dataframe()
    assert list(df.columns) == ["aptamer", "protein", "label"]
    assert df.shape[0] > 0


@pytest.mark.parametrize(
    "split",
    [None, "train", "test"],
)
def test_load_li2014_return_x_y(split):
    """return_X_y=True splits into a MoleculeLoader X and a DataFrame y."""
    X, y = load_li2014(split=split, return_X_y=True)

    assert isinstance(X, MoleculeLoader)
    assert isinstance(y, pd.DataFrame)

    X_df = X.to_dataframe()
    assert list(X_df.columns) == ["aptamer", "protein"]
    assert list(y.columns) == ["label"]
    assert len(X_df) == len(y) > 0


def test_load_li2014_concatenation():
    """split=None concatenates the train and test splits."""
    X_all, y_all = load_li2014(split=None, return_X_y=True)
    X_train, y_train = load_li2014(split="train", return_X_y=True)
    X_test, y_test = load_li2014(split="test", return_X_y=True)

    n_all = len(X_all.to_dataframe())
    n_train = len(X_train.to_dataframe())
    n_test = len(X_test.to_dataframe())

    assert n_all == n_train + n_test
    assert len(y_all) == len(y_train) + len(y_test)
