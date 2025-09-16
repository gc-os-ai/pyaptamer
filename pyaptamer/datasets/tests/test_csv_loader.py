__author__ = "satvshr"

import numpy as np
from sklearn.utils import Bunch

from pyaptamer.datasets._loaders._csv_loader import load_csv_dataset

DATASET_NAME = "train_li2014"
TARGET_COL = "label"


def test_load_csv_return_x_y():
    """
    When return_x_y=True the loader should return two numpy arrays (X, y).
    """
    X, y = load_csv_dataset(DATASET_NAME, target_col=TARGET_COL, return_x_y=True)

    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert X.shape[0] == y.shape[0], "Number of samples in X and y must match"
    assert X.ndim == 2, "X should be a 2D array (n_samples, n_features)"


def test_load_csv_returns_bunch():
    """
    When return_x_y=False the loader should return a sklearn.utils.Bunch-like object
    containing data, target and frame.
    """
    bunch = load_csv_dataset(DATASET_NAME, target_col=TARGET_COL, return_x_y=False)

    assert isinstance(bunch, Bunch), "Returned object should be a sklearn.utils.Bunch"
    assert bunch.data.shape[0] == len(bunch.frame), (
        "data length must match number of rows in frame"
    )
    assert bunch.target_name == TARGET_COL
