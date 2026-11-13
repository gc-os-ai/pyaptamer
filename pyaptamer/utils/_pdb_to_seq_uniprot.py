import io

import pandas as pd
import requests
from Bio import SeqIO

_REQUEST_TIMEOUT = 30  # seconds


def pdb_to_seq_uniprot(pdb_id, return_type="list"):
    """
    Retrieve the canonical UniProt amino-acid sequence for a given PDB ID.

    Parameters
    ----------
    pdb_id : str
        PDB ID (e.g., '1a3n').
    return_type : {'list', 'pd.df'}, optional, default='list'
        Format of returned value:

          - ``'list'`` : list with one amino-acid sequence
          - ``'pd.df'`` : pandas.DataFrame with a single column ['sequence']

    Returns
    -------
    list of str or pandas.DataFrame
        Depending on ``return_type``.

    Raises
    ------
    ValueError
        If `return_type` is invalid, or no UniProt mapping exists for `pdb_id`.
    """
    # validate before doing any network request
    if return_type not in ("list", "pd.df"):
        raise ValueError("`return_type` must be either 'list' or 'pd.df'")

    pdb_id = pdb_id.lower()

    mapping_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    mapping_resp = requests.get(mapping_url, timeout=_REQUEST_TIMEOUT)
    mapping_resp.raise_for_status()
    mapping_data = mapping_resp.json()

    uniprot_ids = list(mapping_data.get(pdb_id, {}).get("UniProt", {}).keys())
    if not uniprot_ids:
        raise ValueError(f"No UniProt mapping found for PDB ID '{pdb_id}'")

    uniprot_id = uniprot_ids[0]

    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    fasta_resp = requests.get(fasta_url, timeout=_REQUEST_TIMEOUT)
    fasta_resp.raise_for_status()
    fasta_data = fasta_resp.text

    record = next(SeqIO.parse(io.StringIO(fasta_data), "fasta"))
    sequence = str(record.seq)

    df = pd.DataFrame({"sequence": [sequence]})

    if return_type == "list":
        return df["sequence"].tolist()
    return df.reset_index(drop=True)
