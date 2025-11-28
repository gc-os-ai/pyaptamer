__author__ = "satvshr"

import pandas as pd
import pytest

from pyaptamer.datasets._loaders._li2014 import (
    load_test_li2014,
    load_train_li2014,
)


@pytest.mark.parametrize(
    "loader",
    [load_train_li2014, load_test_li2014],
)
def test_loader_li2014(loader):
    """
    The loader should return a tuple (X, y) where:
    - X is a DataFrame
    - y is a Series
    - they have matching lengths
    """
    X, y = loader()

    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y should be a pandas Series"

    assert len(X) == len(y), "X and y must have the same number of rows"
    assert X.shape[0] > 0, "X should not be empty"
    assert y.shape[0] > 0, "y should not be empty"
