__author__ = "satvshr"

import pandas as pd
import pytest

from pyaptamer.datasets._loaders._li2014 import load_li2014


@pytest.mark.parametrize(
    "split",
    [None, "train", "test"],
)
def test_load_li2014(split):
    """
    Test that load_li2014 returns a tuple (X, y) where:
    - X is a DataFrame
    - y is a DataFrame
    - they have matching lengths
    - they are non-empty

    Parameters
    ----------
    split : {None, 'train', 'test'}
        Split to load and test.
    """
    X, y = load_li2014(split=split)

    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"
    assert isinstance(y, pd.DataFrame), "y should be a pandas DataFrame"

    assert len(X) == len(y), "X and y must have the same number of rows"
    assert X.shape[0] > 0, "X should not be empty"
    assert y.shape[0] > 0, "y should not be empty"


def test_load_li2014_concatenation():
    """
    Test that loading with split=None returns the concatenation of train and test.
    """
    X_all, y_all = load_li2014(split=None)
    X_train, y_train = load_li2014(split="train")
    X_test, y_test = load_li2014(split="test")

    # Total rows should equal sum of train and test
    assert len(X_all) == len(X_train) + len(X_test), (
        "Concatenated data should have rows equal to train + test"
    )
    assert len(y_all) == len(y_train) + len(y_test), (
        "Concatenated labels should have length equal to train + test"
    )
