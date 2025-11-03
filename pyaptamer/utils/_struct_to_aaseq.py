__author__ = "satvshr"
__all__ = ["struct_to_aaseq"]

import pandas as pd
from Bio.PDB.Polypeptide import PPBuilder


def struct_to_aaseq(structure, return_type="list"):
    """
    Extract amino-acid sequences from a Biopython Structure.

    Parameters
    ----------
    structure :
        Bio.PDB.Structure.Structure object (e.g. produced by PDBParser).
    return_type : {'pd.df', 'list'}, optional, default='list'
        - ``'pd.df'`` : return a pandas.DataFrame with exactly two columns
          (in this order): ``'chain'`` and ``'sequence'``. Each row corresponds
          to one peptide built by PPBuilder. The DataFrame uses the default
          integer index (0..n-1). If no peptides are found an empty DataFrame
          with columns ``['chain','sequence']`` is returned.
        - ``'list'`` : return a Python list of sequence strings (one entry per
          peptide). The order is the same as the DataFrame would be produced:
          chains are iterated in structure.get_chains() order and peptides for
          each chain are appended in PPBuilder order. If no peptides are found
          an empty list is returned.

    Returns
    -------
    pandas.DataFrame or list
        DataFrame with columns ``'chain'`` and ``'sequence'``
        if ``return_type=='pd.df'``, otherwise a list of sequence strings.

    Raises
    ------
    ValueError
        If ``return_type`` is not one of ``{'pd.df','list'}``.
    """
    ppb = PPBuilder()
    records = []
    for chain in structure.get_chains():
        peptides = ppb.build_peptides(chain)
        for pep in peptides:
            records.append({"chain": chain.id, "sequence": str(pep.get_sequence())})

    # Ensure consistent column order and empty-structure behavior
    df = pd.DataFrame.from_records(records, columns=["chain", "sequence"])

    if return_type == "pd.df":
        return df
    elif return_type == "list":
        return df["sequence"].tolist()
    else:
        raise ValueError("return_type must be one of {'pd.df', 'list'}")
