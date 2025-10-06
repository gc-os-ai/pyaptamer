__author__ = "satvshr"
__all__ = ["pdb_to_aaseq"]

import io
import os

import pandas as pd
import requests
from Bio import SeqIO

from ._pdb_to_struct import pdb_to_struct
from ._struct_to_aaseq import struct_to_aaseq


def pdb_to_aaseq(pdb_file_path, return_type="list", use_uniprot=False, pdb_id=None):
    """
    Extract amino-acid sequences from a PDB file.

    Tries SEQRES records first (full deposited sequence).
    Falls back to using the package's pdb -> Structure -> sequences converters
    if SEQRES records are not present. Optionally, retrieves canonical UniProt
    sequence for the PDB ID.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
        Path to a PDB file.
    return_type : {'list', 'pd.df'}, optional, default='list'
        Format of returned value:
          - ``'list'`` : list of amino acid strings (one per chain / polypeptide)
          - ``'pd.df'`` : pandas.DataFrame with a single column ``'sequence'``.
            Rows are indexed 0..n-1 (no chain identifiers).
    use_uniprot : bool, optional, default=False
        If True, fetches the UniProt sequence using the PDB ID.
        Requires the ``pdb_id`` argument to be set.
    pdb_id : str, optional
        PDB ID (e.g., ``'1a3n'``) required if ``use_uniprot=True``.

    Returns
    -------
    list of str or pandas.DataFrame
        Depending on ``return_type``. If ``'list'``, returns a Python list of
        sequence strings (one element per chain / polypeptide). If ``'pd.df'``,
        returns a DataFrame with a single column ``'sequence'`` and a default
        integer index (no chain IDs).

    Raises
    ------
    FileNotFoundError
        If the given ``pdb_file_path`` does not exist.
    ValueError
        If ``return_type`` is not one of the supported values, or if
        ``use_uniprot=True`` but no mapping / fasta could be retrieved.
    """
    pdb_path = os.fspath(pdb_file_path)
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    sequences = []

    # Try SEQRES records first
    with open(pdb_path) as handle:
        seqres_records = list(SeqIO.parse(handle, "pdb-seqres"))

    if seqres_records:
        for record in seqres_records:
            sequences.append(str(record.seq))
    else:
        # Fall back to using pdb_to_struct + struct_to_aaseq helpers
        structure = pdb_to_struct(pdb_path)
        sequences = struct_to_aaseq(structure)

    if len(sequences) == 0:
        raise ValueError(f"No sequences could be extracted from PDB file: {pdb_path}")

    if use_uniprot:
        if not pdb_id:
            raise ValueError("`pdb_id` must be provided when use_uniprot=True")

        pdb_id = pdb_id.lower()
        mapping_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
        mapping_resp = requests.get(mapping_url, timeout=10)
        mapping_resp.raise_for_status()
        mapping_data = mapping_resp.json()
        uniprot_ids = list(mapping_data.get(pdb_id, {}).get("UniProt", {}).keys())

        if not uniprot_ids:
            raise ValueError(f"No UniProt mapping found for PDB ID '{pdb_id}'")

        uniprot_id = uniprot_ids[0]

        fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        fasta_resp = requests.get(fasta_url, timeout=10)
        fasta_resp.raise_for_status()
        fasta_data = fasta_resp.text

        record = next(SeqIO.parse(io.StringIO(fasta_data), "fasta"))
        sequences = [str(record.seq)]

    if return_type == "list":
        return sequences
    elif return_type == "pd.df":
        df = pd.DataFrame({"sequence": sequences})
        return df
    else:
        raise ValueError("`return_type` must be either 'list' or 'pd.df'")
