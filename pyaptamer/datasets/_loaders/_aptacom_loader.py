__author__ = "rpgv"
__all__ = ["load_aptacom_full", "load_aptacom_xy"]

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


def filter_columns(ds, columns=None):
    """ " Selects columns to keep on dataset
    Parameters:
    -----------
        ds: pd dataframe, required
        Pandas dataframe to filter
        columns: list, optional, default=None
        If empty returns entire AptaCom dataset, otherwise
        returns only the selected columns from the
        AptaCom dataset
    Returns:
    --------
        object: pandas dataframe object with
        the selected columns
    """

    if columns is not None:
        ds = ds[columns]
    return ds


def prepare_xy(ds):
    """ " Prepares dataset for usage as training data
    Parameters:
    -----------
    ds: pandas dataframe, required

    Returns:
    --------
    Pandas dataframe object processed for training
    with columns "aptamer_sequence", "target_sequence",
    "new_affinity" and a total of 1061 columns
    """
    ds.dropna(
        subset=["aptamer_sequence", "target_sequence", "new_affinity"], inplace=True
    )
    ds = ds[["aptamer_sequence", "target_sequence", "new_affinity"]]
    return ds


def load_aptacom_full(select_columns=None):
    """Loads a AptaCom dataset from hugging face
    with customizable options.

    Parameters:
    -----------
    select_columns: list, optional, default=None
        A list used to filter the columns dataset features.
        Defaults to empty, which returns the complete dataset.
        Column names:
        ['reference',
        'aptamer_chemistry',
        'aptamer_name',
        'target_name',
        'aptamer_sequence',
        'origin',
        'target_chemistry',
        'external_id',
        'target_sequence',
        'new_affinity']

    Returns:
    --------
        object: A pandas dataframe with 5556 rows in total.
        The returned object contains the dataset, possibly
        filtered with different columns.
    """

    aptacom = load_hf_dataset("AptaCom", store=False)
    dataset = filter_columns(aptacom, columns=select_columns)

    return dataset


def load_aptacom_xy(return_X_y=False):
    """Loads Aptacom dataset for training

    Parameters:
    ----------
    return_X_y: bool, optional, default = False
        If true returns X (aptamer and target sequence)
        and y (new_affinity) otherwise returns a
        pandas dataframe containing the three columns

    Returns:
    --------
    Either a pandas dataframe with three columns
    or two pandas dataframe objects with two and one
    columns respectively
    """
    aptacom = load_hf_dataset("AptaCom", store=False)
    dataset = prepare_xy(aptacom)
    if return_X_y:
        X = dataset[["aptamer_sequence", "target_sequence"]]
        y = dataset[["new_affinity"]]
        return X, y
    return dataset
