__author__ = "satvshr"

import numpy as np
import pytest

from pyaptamer.pseaac import PSeAAC
from pyaptamer.pseaac._props import aa_props
from pyaptamer.pseaac.tests._props import solution


def test_normalized_values():
    """
    Test that normalized property matrix matches expected normalized values.

    Asserts
    -------
    All normalized property values match the hard-coded normalized matrix
    for each amino acid and property.
    """
    # Get original and normalized property matrices as DataFrames
    original_df = aa_props(type="pandas", normalize=False)
    normalized_df = aa_props(type="pandas", normalize=True)

    # Manually normalize the original matrix (z-score, column-wise,
    # rounded to 3 decimals)
    manual_norm = (original_df - original_df.mean()) / original_df.std(ddof=0)
    manual_norm = manual_norm.round(3)

    # Compare each value
    for aa in original_df.index:
        for prop in original_df.columns:
            assert normalized_df.loc[aa, prop] == manual_norm.loc[aa, prop], (
                f"Mismatch for {aa}, {prop}: "
                f"{normalized_df.loc[aa, prop]} != {manual_norm.loc[aa, prop]}"
            )


@pytest.mark.parametrize(
    "seq,lambda_val",
    [
        ("ACDEFGHIK", 10),  # length 9, lambda_val 10
        ("ACDAA", 5),  # length 5, lambda_val 5
        ("A", 2),  # length 1, lambda_val 2
    ],
)
def test_pseaac_transform_sequence_too_short(seq, lambda_val):
    """
    Test that the PSeAAC transform method raises an error for protein sequences of
    length smaller or equal to lambda_val.
    """
    p = PSeAAC(lambda_val=lambda_val)
    with pytest.raises(ValueError, match="Protein sequence is too short"):
        p.transform(seq)


@pytest.mark.skip(reason="Pending issue #34")
@pytest.mark.parametrize(
    "seq,expected_vector",
    [
        (
            "ACDFFKKIIKKLLMMNNPPQQQRRRRIIIIRRR",
            solution,
        )
    ],
)
def test_pseaac_vectorization(seq):
    """
    Test that the PSeAAC vectorization works without throwing an error.

    Parameters
    ----------
    seq : str
        Protein sequence to transform.

    Asserts
    ----------
    Output vector after PSeAAC is a numpy array.
    """
    p = PSeAAC()
    pv = p.transform(seq)

    assert isinstance(pv, np.ndarray)
