__author__ = "satvshr"

import pandas as pd

from pyaptamer.datasets._loaders._1gnh import _load_1gnh_structure
from pyaptamer.utils._struct_to_aaseq import struct_to_aaseq


def test_struct_to_aaseq():
    """
    Test that `struct_to_aaseq` correctly converts a Biopython Structure
    into both a pandas DataFrame and a list of sequences.

    Asserts:
        - No exception is raised when calling the function.
        - The DataFrame return value has columns "chain" and "sequence" (in that order).
        - The list return value is a list of strings matching the DataFrame sequences.
        - Each 'sequence' value is a non-empty string.
        - Each 'chain' value is a non-empty string.
    """
    structure = _load_1gnh_structure()

    # Request DataFrame explicitly (columns should be exactly ['chain','sequence'])
    df = struct_to_aaseq(structure, return_type="pd.df")

    assert isinstance(df, pd.DataFrame), "Return value should be a pandas DataFrame"
    assert list(df.columns) == ["chain", "sequence"], (
        "DataFrame must have columns ['chain','sequence']"
    )
    assert not df.empty, "Returned DataFrame should not be empty"

    for chain, seq in zip(df["chain"], df["sequence"], strict=False):
        assert isinstance(chain, str) and len(chain) > 0, (
            "Each chain id should be a non-empty string"
        )
        assert isinstance(seq, str) and len(seq) > 0, (
            "Each sequence should be a non-empty string"
        )

    seq_list = struct_to_aaseq(structure)  # default list
    assert isinstance(seq_list, list), "Default return should be a list of sequences"
    assert len(seq_list) == len(df), "List length must match number of DataFrame rows"
    assert seq_list == df["sequence"].tolist(), (
        "List sequences must match DataFrame 'sequence' column"
    )
