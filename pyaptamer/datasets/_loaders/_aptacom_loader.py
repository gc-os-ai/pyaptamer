# file: aptacom_loader.py

__author__ = "rpgv"
__all__ = ["load_aptacom_full", "load_aptacom_x_y"]

from pyaptamer.datasets._loaders._hf_loader import load_hf_dataset

filter_map = {
    "protein_target": ("target_chemistry", ["Protein", "peptide"]),
    "small_target": (
        "target_chemistry",
        ["Small Organic", "Small Molecule", "Molecule"],
    ),
    "dna_apt": (
        "aptamer_chemistry",
        [
            "DNA",
            "L-DNA",
            "ssDNA",
            "2',4'-BNA/LNA-DNA",
            "5-uracil-modified-DNA",
            "dsDNA",
        ],
    ),
    "rna_apt": (
        "aptamer_chemistry",
        [
            "RNA",
            "2'-F-RNA",
            "2'-NH2-RNA",
            "L-RNA",
            "2'-O-Me-RNA",
            "ssRNA",
            "2'-fluoro/amino-RNA",
            "2'-fluoro-RNA",
            "2'-amino-RNA",
            "2'-fluoro/O-Me-RNA",
            "5-uracil-modified-RNA",
            "4'-thio-RNA",
        ],
    ),
}


def _filter_columns(ds, columns=None):
    """
    Select a subset of columns from a pandas DataFrame.

    Parameters
    ----------
    ds : pandas.DataFrame
        Input DataFrame to filter.
    columns : list[str] or None, optional
        Column names to keep. If None, returns the input DataFrame unchanged.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only the requested columns (or the original
        DataFrame if `columns` is None).

    """
    if columns is not None:
        ds = ds[columns]
    return ds


def prepare_x_y(ds):
    """
    Prepare dataset for training by selecting required columns and dropping rows with missing values.

    This function:
    - Drops rows with missing values in the columns
      "aptamer_sequence", "target_sequence", and "new_affinity".
    - Keeps only those three columns.

    Parameters
    ----------
    ds : pandas.DataFrame
        Input DataFrame containing at least the columns
        "aptamer_sequence", "target_sequence", and "new_affinity".

    Returns
    -------
    pandas.DataFrame
        A DataFrame with exactly the columns:
        ["aptamer_sequence", "target_sequence", "new_affinity"],
        and with rows containing no missing values in those columns.

    """
    ds.dropna(
        subset=["aptamer_sequence", "target_sequence", "new_affinity"], inplace=True
    )
    ds = ds[["aptamer_sequence", "target_sequence", "new_affinity"]]
    return ds


def load_aptacom_full(select_columns=None):
    """
    Load the AptaCom dataset from Hugging Face, with optional column selection.

    Parameters
    ----------
    select_columns : list[str] or None, optional
        List of column names to retain. If None, returns the full dataset.

        Available columns include (subject to upstream changes):
        [
            'reference',
            'aptamer_chemistry',
            'aptamer_name',
            'target_name',
            'aptamer_sequence',
            'origin',
            'target_chemistry',
            'external_id',
            'target_sequence',
            'new_affinity'
        ]

    Returns
    -------
    pandas.DataFrame
        The loaded dataset, optionally filtered to the selected columns.

    """
    aptacom = load_hf_dataset("AptaCom", store=False)
    dataset = _filter_columns(aptacom, columns=select_columns)
    return dataset


def load_aptacom_x_y(return_X_y=False):
    """
    Load the AptaCom dataset prepared for model training.

    Depending on `return_X_y`, returns either a single DataFrame containing
    the features and target, or a tuple of (X, y) DataFrames.

    Parameters
    ----------
    return_X_y : bool, optional
        If True, return a tuple `(X, y)` where:
          - `X` has columns ["aptamer_sequence", "target_sequence"]
          - `y` has column ["new_affinity"]
        If False (default), return a single DataFrame with all three columns.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, pandas.DataFrame]
        - If `return_X_y` is False: a DataFrame with columns
          ["aptamer_sequence", "target_sequence", "new_affinity"].
        - If `return_X_y` is True: a tuple `(X, y)` where `X` contains the two
          feature columns and `y` contains the target column.
    """
    aptacom = load_hf_dataset("AptaCom", store=False)
    dataset = prepare_x_y(aptacom)
    if return_X_y:
        X = dataset[["aptamer_sequence", "target_sequence"]]
        y = dataset[["new_affinity"]]
        return X, y
    return dataset