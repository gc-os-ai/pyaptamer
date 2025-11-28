__author__ = "satvshr"
__all__ = ["load_train_li2014", "load_test_li2014"]
import os

import pandas as pd


def load_train_li2014():
    """
    Load the Li 2014 training dataset.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Labels/target.
    """
    # Path relative to this file
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "train_li2014.csv")
    )

    df = pd.read_csv(path)

    # Basic assumption: last column is the label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def load_test_li2014():
    """
    Load the Li 2014 test dataset.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Labels/target.
    """
    # Path relative to this file
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "test_li2014.csv")
    )

    df = pd.read_csv(path)

    # Basic assumption: last column is the label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y
