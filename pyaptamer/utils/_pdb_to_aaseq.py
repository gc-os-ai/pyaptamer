__author__ = "satvshr"
__all__ = ["pdb_to_aaseq"]

import io
import os

import pandas as pd
import requests
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder


def pdb_to_aaseq(pdb_file_path, return_type="list", use_uniprot=False, pdb_id=None):
    """
    Extract amino-acid sequences from a PDB file.

    Tries SEQRES records first (full deposited sequence).
    Falls back to ATOM coordinates if SEQRES missing.
    Optionally, retrieves canonical UniProt sequence for the PDB ID.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
        Path to a PDB file.
    return_type : {'list', 'pd.df'}, optional, default='list'
        Format of returned value:
          - 'list' : list of amino acid strings (one per chain)
          - 'pd.df' : DataFrame with chain id and sequence
    use_uniprot : bool, optional, default=False
        If True, fetches the UniProt sequence using the PDB ID.
        Requires `pdb_id` argument to be set.
    pdb_id : str, optional
        PDB ID (e.g., '1a3n') required if use_uniprot=True.

    Returns
    -------
    list of str or pandas.DataFrame
        Depending on `return_type`.
    """

    pdb_path = os.fspath(pdb_file_path)
    sequences, chains = [], []

    # Try SEQRES records first
    with open(pdb_path) as handle:
        seqres_records = list(SeqIO.parse(handle, "pdb-seqres"))

    if seqres_records:
        for record in seqres_records:
            sequences.append(str(record.seq))
            chains.append(getattr(record, "id", None))
    else:
        # Fall back to ATOM records
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb", pdb_path)
        ppb = PPBuilder()
        for model in structure:
            for chain in model:
                peptides = ppb.build_peptides(chain)
                if not peptides:
                    continue
                seq = "".join(str(p.get_sequence()) for p in peptides)
                sequences.append(seq)
                chains.append(chain.id)

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
        chains = [uniprot_id]

    if return_type == "list":
        return sequences
    elif return_type == "pd.df":
        df = pd.DataFrame({"chain": chains, "sequence": sequences})
        df = df.set_index("chain")
        return df
    else:
        raise ValueError("`return_type` must be either 'list' or 'pd.df'")
