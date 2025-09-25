__author__ = "satvshr"
__all__ = ["fasta_to_aaseq"]

import os
from io import StringIO

import pandas as pd
from Bio import SeqIO

from pyaptamer.utils.hf_to_dataset import hf_to_dataset


def fasta_to_aaseq(fasta_path, return_df=False):
    """
    Extract sequences from a FASTA file, Hugging Face dataset, or URL.

    Parameters
    ----------
    ``fasta_path`` : str or os.PathLike
        Input source for FASTA sequences. Can be:

          - Local file path (absolute or relative) located in
          ``pyaptamer/datasets/data``.
          - Hugging Face Hub reference.

    ``return_df`` : bool, default=False
        If True, return a pandas DataFrame with index = FASTA record id and
        a single column:

          - ``sequence`` : str
              Sequence string (amino acid or nucleotide).

        If False, return a list of sequence strings.

    Returns
    -------
    list of str or pandas.DataFrame

        - If ``return_df=False``: list of sequences (str).
        - If ``return_df=True``: DataFrame indexed by record id with a
          ``sequence`` column. Returns an empty DataFrame with index name
          ``id`` if no records are found.

    """
    path = os.path.join(os.path.dirname(__file__), "..", "data", fasta_path)
    if os.path.exists(path):
        fasta_handle = path
    else:
        dataset = hf_to_dataset(fasta_path)
        content = "\n".join(dataset[:]["text"])
        fasta_handle = StringIO(content)

    records = [
        {"id": record.id, "sequence": str(record.seq)}
        for record in SeqIO.parse(fasta_handle, "fasta")
    ]

    if return_df:
        df = pd.DataFrame(records).set_index("id")
        return df

    return [r["sequence"] for r in records]
