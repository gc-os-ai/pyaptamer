__author__ = ["satvshr"]
__all__ = ["load_csv_dataset"]

import os

import pandas as pd
from sklearn.utils import Bunch


def load_csv_dataset(name, target_col, return_X_y=False):
    """
    Load a dataset from a CSV file in a sklearn-like format.

    Parameters
    ----------
    name : str
        Name of the dataset (file basename without `.csv`) located in the
        package `dataset/data/` directory.
    target_col : str
        Column name in the CSV to use as the target variable.
    return_X_y : bool, optional, default=False
        If True, return (X, y) as NumPy arrays. If False, return a sklearn.utils.Bunch
        with attributes similar to sklearn dataset loaders.

    Returns
    -------
    sklearn.utils.Bunch or tuple of np.ndarray
        If `return_X_y` is False, returns a Bunch with fields:
            - data: ndarray of shape (n_samples, n_features)
            - target: ndarray of shape (n_samples,)
            - frame: pandas.DataFrame (the loaded DataFrame)
            - feature_names: list[str] (column names used as features)
            - target_name: str (the name of the target column)
            - filename: str (resolved path to the CSV file)
        If `return_X_y` is True, returns (X, y) where:
            - X : ndarray of shape (n_samples, n_features) built by dropping the target
            column
            - y : ndarray of shape (n_samples,) from the target column
    """
    path = os.path.relpath(
        os.path.join(os.path.dirname(__file__), "..", "data", f"{name}.csv")
    )

    df = pd.read_csv(path)

    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()

    if return_X_y:
        return X, y

    bunch = Bunch(
        data=X,
        target=y,
        frame=df,
        feature_names=list(df.columns.drop(target_col)),
        target_name=target_col,
        filename=path,
    )
    return bunch
