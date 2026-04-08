import io
from typing import Union

import pandas as pd
import requests
from Bio import SeqIO


def pdb_to_seq_uniprot(pdb_id: str, return_type: str = "list") -> Union[list[str], pd.DataFrame]:
    """
    Retrieve the canonical UniProt amino-acid sequence for a given PDB ID.

    Parameters
    ----------
    pdb_id : str
        PDB ID (e.g., '1a3n').
    return_type : {'list', 'pd.df'}, optional, default='list'
        Format of returned value:

          - ``'list'`` : list with one amino-acid sequence string
          - ``'pd.df'`` : pandas.DataFrame with a single column ['sequence']

    Returns
    -------
    list[str] or pandas.DataFrame
        Depending on ``return_type``:
          - If ``'list'``: Python list containing the sequence string.
          - If ``'pd.df'``: DataFrame with column ``['sequence']``.

    Raises
    ------
    ValueError
        If no UniProt mapping is found for the given PDB ID, or if
        ``return_type`` is invalid.
    requests.exceptions.HTTPError
        If the connection to the PDBe or UniProt API fails.

    Examples
    --------
    >>> from pyaptamer.utils import pdb_to_seq_uniprot
    >>> seq = pdb_to_seq_uniprot("1a3n", return_type="list")
    >>> print(seq[0][:10])  # Print first 10 amino acids
    VLSPADKTNV
    """
    pdb_id = pdb_id.lower()

    # 1. Fetch UniProt mapping from EBI PDBe API
    mapping_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    mapping_resp = requests.get(mapping_url)
    mapping_resp.raise_for_status()
    mapping_data = mapping_resp.json()

    # Extract the first available UniProt ID
    uniprot_ids = list(mapping_data.get(pdb_id, {}).get("UniProt", {}).keys())
    if not uniprot_ids:
        raise ValueError(f"No UniProt mapping found for PDB ID '{pdb_id}'")

    uniprot_id = uniprot_ids[0]

    # 2. Fetch the canonical FASTA sequence from UniProt
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    fasta_resp = requests.get(fasta_url)
    fasta_resp.raise_for_status()
    fasta_data = fasta_resp.text

    # 3. Parse the FASTA response and extract the sequence
    record = next(SeqIO.parse(io.StringIO(fasta_data), "fasta"))
    sequence = str(record.seq)

    df = pd.DataFrame({"sequence": [sequence]})

    if return_type == "list":
        return df["sequence"].tolist()
    elif return_type == "pd.df":
        return df.reset_index(drop=True)
    else:
        raise ValueError("`return_type` must be either 'list' or 'pd.df'")