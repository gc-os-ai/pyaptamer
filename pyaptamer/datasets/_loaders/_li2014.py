__author__ = ["satvshr", "siddharth7113"]
__all__ = ["load_li2014"]
import os

import pandas as pd

from pyaptamer.data.loader import MoleculeLoader


def load_li2014(split=None, return_X_y=False):
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

    The molecule data is returned as a
    :class:`~pyaptamer.data.loader.MoleculeLoader` so it plugs directly into the
    AptaNet transform pipeline, mirroring
    :func:`~pyaptamer.datasets.load_aptacom`.

    Parameters
    ----------
    split : {None, "train", "test"}, optional
        Which split to load. ``None`` (default) concatenates train+test.
    return_X_y : bool, optional
        If True, return a tuple ``(X, y)`` where:
          - ``X`` is a MoleculeLoader over the feature columns
            ["aptamer", "protein"]
          - ``y`` is a DataFrame with the target column ["label"]
        If False (default), return a single MoleculeLoader over all three
        columns ["aptamer", "protein", "label"].

    Returns
    -------
    MoleculeLoader or tuple[MoleculeLoader, pandas.DataFrame]
        - If `return_X_y` is False: a MoleculeLoader over the columns
          ["aptamer", "protein", "label"].
        - If `return_X_y` is True: a tuple ``(X, y)`` where ``X`` is a
          MoleculeLoader over the two feature columns and ``y`` is a DataFrame
          with the target. The target keeps the column name "label" as present
          in the CSV; using a DataFrame for `y` keeps the shape consistent for
          downstream code even when the target is one column.
    """
    if split not in (None, "train", "test"):
        raise ValueError("split must be None, 'train', or 'test'")

    base_path = os.path.join(os.path.dirname(__file__), "..", "data")

    dfs = []

    # Load train split if requested
    if split is None or split == "train":
        train_path = os.path.join(base_path, "train_li2014.csv")
        dfs.append(pd.read_csv(train_path))

    # Load test split if requested
    if split is None or split == "test":
        test_path = os.path.join(base_path, "test_li2014.csv")
        dfs.append(pd.read_csv(test_path))

    # Concatenate splits if both were loaded
    dataset = pd.concat(dfs, ignore_index=True)

    if return_X_y:
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1:]
        return MoleculeLoader(data=X), y
    return MoleculeLoader(data=dataset)
