__author__ = "satvshr"
__all__ = ["load_li2014"]
import os

import pandas as pd


def load_li2014(split=None):
    """
    Load the Li 2014 aptamer–protein interaction dataset.

    The dataset originates from the AptaTrans training data (Li et al., 2014
    as used by the original AptaTrans code).

    Behaviour
    ---------

    - If `split is None` (default) both train and test CSVs are loaded and
      concatenated (train followed by test).
    - If `split == "train"` only the train CSV is loaded.
    - If `split == "test"` only the test CSV is loaded.

    Expected file layout
    --------------------
    The CSVs use three columns (in this order):

    - aptamer : str
        Aptamer sequence (nucleotide sequence using one-letter IUPAC codes).
        Stored as a plain string (e.g. "ACGUA...").
    - protein : str
        Protein sequence (amino-acid one-letter codes), stored as a string
        (e.g. "MKT...").
    - label : int
        Target label. In the CSV this is a numeric target used for supervised
        learning:

          - positive : interacting / binding
          - negative : non-interacting / non-binding

        The loader preserves the original dtype and values (may also hold
        continuous affinity values in other variants).

    Return types
    ------------
    Returns a tuple (X, y) where:

      - X : pandas.DataFrame
          Feature DataFrame containing the two feature columns (aptamer, protein).
      - y : pandas.DataFrame
          Single-column DataFrame containing the target/label (keeps column name
          "label" as present in the CSV). Using a DataFrame for `y` keeps the
          shape consistent for downstream code even when the target is one column.

    Parameters
    ----------
    split : {None, "train", "test"}, optional
        Which split to load. ``None`` (default) concatenates train+test.

    Examples
    --------
    >>> X_all, y_all = load_li2014()  # train + test concatenated
    >>> X_train, y_train = load_li2014("train")
    >>> X_test, y_test = load_li2014("test")
    """
    if split not in (None, "train", "test"):
        raise ValueError("split must be None, 'train', or 'test'")

    base_path = os.path.join(os.path.dirname(__file__), "..", "data")

    dfs_X = []
    dfs_y = []

    # Load train split if requested
    if split is None or split == "train":
        train_path = os.path.join(base_path, "train_li2014.csv")
        train_df = pd.read_csv(train_path)
        dfs_X.append(train_df.iloc[:, :-1])
        dfs_y.append(train_df.iloc[:, -1:])

    # Load test split if requested
    if split is None or split == "test":
        test_path = os.path.join(base_path, "test_li2014.csv")
        test_df = pd.read_csv(test_path)
        dfs_X.append(test_df.iloc[:, :-1])
        dfs_y.append(test_df.iloc[:, -1:])

    # Concatenate splits if both were loaded
    X = pd.concat(dfs_X, ignore_index=True)
    y = pd.concat(dfs_y, ignore_index=True)

    return X, y
