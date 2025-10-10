__author__ = "rpgv"
__all__ = ["load_complete_aptacom", "load_aptacom_for_training"]

import pandas as pd
from datasets import load_dataset

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


def load_complete_aptacom(as_df=False, filter_entries=None):
    """Loads a Hugging Face dataset with customizable
    options.

        Args:
            as_df (bool, optional): If True, returns the
                dataset as a pandas DataFrame.
                    Defaults to False, which returns the dataset as
                    a Hugging Face Dataset object.
            filter_entries (str, optional):  A string used to
                filter the dataset features.
                    The format is a specific key:
                        (protein_target, small_target, dna_apt, rna_apt)
                        filters either by target (protein or small molecule)
                        of by aptamer (dna or rna)
                        Defaults to None,
                            meaning no filtering is applied.

        Returns:
            object: A Hugging Face Dataset object (if as_df is
            False) or a pandas DataFrame (if as_df is True) with
            10 columns in total.
            The returned object contains the dataset, possibly
            filtered depending on the arguments affecting
            the number of rows.

            (
            filter_entries: if protein_target, total of 1110 rows;
            filter_entries: if small_target, total of 573 rows;
            filter_entries: if dna_apt, total of 2395 rows;
            filter_entries: if rna_apt, total of 1152 rows;
            )

        Raises:

            ValueError: if filter_entries is a string but
    contains an invalid format (e.g., empty string)

    """
    aptacom = load_dataset("rpgv/AptaCom")["train"]
    if filter_entries is not None:
        if filter_entries not in filter_map.keys():
            raise ValueError("""Key error\nFilter arguments: protein_target, 
            small_target, dna_apt, rna_apt""")
        else:
            aptacom = aptacom.filter(
                lambda x: x[filter_map[filter_entries][0]]
                in filter_map[filter_entries][1]
            )

    try:
        if as_df:
            dataset = pd.DataFrame().from_dict(aptacom)
        else:
            dataset = aptacom.with_format("pandas")
    except Exception as e:
        print(f"""Error: {e}\n'as_df' parameter expects
                          a boolean (True or False)""")

    return dataset


def load_aptacom_for_training(as_df=False, filter_entries=None, tagret_id=False):
    """Loads a Hugging Face dataset with customizable
    options.

        Args:
            as_df (bool, optional): If True, returns the
                dataset as a pandas DataFrame.
                    Defaults to False, which returns the dataset as
                    a Hugging Face Dataset object.
            filter_entries (str, optional):  A string used to
                filter the dataset features.
                    The format is a specific key:
                        (protein_target, small_target, dna_apt, rna_apt)
                        filters either by target (protein or small molecule)
                        of by aptamer (dna or rna)
                        Defaults to None,
                            meaning no filtering is applied.
            tagret_id (bool, optional): If True, includes the
                'external_id' column in the dataset.
                    Defaults to False.  If False, the 'external_id' column is
                    excluded.

        Returns:
            object: A Hugging Face Dataset object (if as_df is
            False) or a pandas DataFrame (if as_df is True).
            The returned object contains the dataset, possibly
            filtered and with or without the 'external_id'
            column, depending on the arguments.

            (
            tagret_id: if True, total of 3 columns;
            filter_entries: if protein_target, total of 1110 rows;
            filter_entries: if small_target, total of 573 rows;
            filter_entries: if dna_apt, total of 2395 rows;
            filter_entries: if rna_apt, total of 1152 rows;
            )

        Raises:

            ValueError: if filter_entries is a string but
    contains an invalid format (e.g., empty string)

    """
    aptacom = load_dataset("rpgv/AptaCom")["train"]
    if filter_entries is not None:
        if filter_entries not in filter_map.keys():
            raise ValueError("""Key error\nFilter arguments: protein_target, 
            small_target, dna_apt, rna_apt""")
        else:
            aptacom = aptacom.filter(
                lambda x: x[filter_map[filter_entries][0]]
                in filter_map[filter_entries][1]
            )
    try:
        if tagret_id:
            aptacom = aptacom.map(
                remove_columns=[
                    "reference",
                    "aptamer_chemistry",
                    "aptamer_name",
                    "target_name",
                    "origin",
                    "target_chemistry",
                    "new_affinity",
                ]
            )
        else:
            aptacom = aptacom.map(
                remove_columns=[
                    "reference",
                    "aptamer_chemistry",
                    "aptamer_name",
                    "target_name",
                    "origin",
                    "target_chemistry",
                    "external_id",
                    "new_affinity",
                ]
            )
    except Exception as e:
        print(f"""Error: {e}\n'target_id' parameter expects
                          a boolean (True or False)""")

    try:
        if as_df:
            dataset = pd.DataFrame().from_dict(aptacom)
        else:
            dataset = aptacom.with_format("pandas")
    except Exception as e:
        print(f"""Error: {e}\n'as_df' parameter expects
                          a boolean (True or False)""")

    return dataset
