__author__ = "satvshr"

import pandas as pd

from pyaptamer.datasets import load_1gnh_structure
from pyaptamer.utils import struct_to_aaseq


def test_struct_to_aaseq():
    """
    Test that `struct_to_aaseq` correctly converts a Biopython Structure
    into a pandas DataFrame with columns ['chain', 'sequence'].

    Asserts:
        - No exception is raised when calling the function.
        - The return value is a pandas.DataFrame.
        - The DataFrame has columns "chain" and "sequence" (in that order).
        - Each 'sequence' value is a non-empty string.
        - Each 'chain' value is a non-empty string.
    """
    structure = load_1gnh_structure()

    df = struct_to_aaseq(structure)

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
