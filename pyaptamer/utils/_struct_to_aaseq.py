__author__ = "satvshr"
__all__ = ["struct_to_aaseq"]

import pandas as pd
from Bio.PDB.Polypeptide import PPBuilder


def struct_to_aaseq(structure):
    """
    Extract amino-acid sequences from a Biopython Structure and return a DataFrame
    containing the chain identifiers.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        A Biopython Structure object (e.g. produced by PDBParser).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
          - ``chain`` : Chain identifier (single-character string)
          - ``sequence`` : One-letter amino-acid sequence string for the peptide

        Each row corresponds to one peptide built by ``PPBuilder``. If a chain
        contains multiple peptide fragments (due to gaps) there will be multiple
        rows with the same ``chain`` value.
    """
    ppb = PPBuilder()
    records = []
    for chain in structure.get_chains():
        peptides = ppb.build_peptides(chain)
        for pep in peptides:
            records.append({"chain": chain.id, "sequence": str(pep.get_sequence())})

    df = pd.DataFrame.from_records(records, columns=["chain", "sequence"])
    return df
