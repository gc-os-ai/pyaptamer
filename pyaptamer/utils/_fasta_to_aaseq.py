__author__ = "satvshr"
__all__ = ["fasta_to_aaseq"]

import os
from io import StringIO

import pandas as pd
from Bio import SeqIO

from pyaptamer.utils._hf_to_dataset import hf_to_dataset


def fasta_to_aaseq(fasta_path, return_df=False):
    """
    Extract sequences from a FASTA file, Hugging Face dataset (FASTA-as-text),
    or URL.

    Parameters
    ----------
    fasta_path : str
        Input location for FASTA content. Three input types are supported,
        tried in this order:

        1. Local filesystem path (absolute or relative to current working dir).
        2. Hugging Face dataset identifier. If the string is not an existing
           local path, the function passes it to ``hf_to_dataset`` and expects
           the returned dataset to contain a column named ``"text"`` where each
           row is a FASTA line. In other words, the dataset must be a FASTA file
           that was uploaded as a text dataset on the Hub. Other schemas are not
           supported.
        3. URL. If ``fasta_path`` is a direct URL pointing to a text file
           (e.g. a FASTA file hosted on a website), it is also handled via
           ``hf_to_dataset``. As with Hugging Face datasets, the content is
           expected to be plain FASTA text.

    return_df : bool, default=False
        If ``False`` (default) return a Python list of sequences (one str per
        FASTA record, in the order they appear). If ``True`` return a
        ``pandas.DataFrame`` indexed by the FASTA record id with a single column
        named ``"sequence"``.

    Returns
    -------
    list[str] or pandas.DataFrame

        - If ``return_df=False``: list of sequence strings.
        - If ``return_df=True``: DataFrame with FASTA record ids as index
          and one column ``"sequence"``.

    Examples
    --------

    - Local FASTA file:
      ``fasta_to_aaseq("my_sequences.fasta")``
    - Hugging Face dataset identifier (dataset must contain FASTA text in a
      column named "text"):
      ``fasta_to_aaseq("username/fasta_dataset", return_df=True)``
    - Direct URL to a FASTA file:
      ``fasta_to_aaseq("https://example.org/my_sequences.fasta")``

    """
    if os.path.exists(fasta_path):
        fasta_handle = fasta_path
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
