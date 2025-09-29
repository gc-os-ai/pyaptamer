__author__ = "rpgv"
__all__ = ["load_aptacom"]

import pandas as pd
from datasets import load_dataset


def load_aptacom(as_df=False, filter_entries=None):
    """
    Loads complete aptacom dataset from hugging face datasets.

    Parameters
    ----------
    as_df: (bool) Requires pandas compatible format; converts
    dataset into pandas dataframe

    filter(str): Allows filtering dataset by:
        - Target chemistry: (protein_target, small_target)
        - Aptamer chemistry: (dna_apt, rna_apt)

    Returns
    -------
    dataset: Hugging face dataset in a pandas compatible format.
    """
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

    if as_df:
        dataset = pd.DataFrame().from_dict(aptacom)
    else:
        dataset = aptacom.with_format("pandas")

    return dataset
