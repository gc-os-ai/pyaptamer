__author__ = "satvshr"

import numpy as np
import pytest

from pyaptamer.aptanet.pseaac import AptaNetPSeAAC
from pyaptamer.aptanet.pseaac._props import aa_props
from pyaptamer.aptanet.pseaac.tests._props import solution

vector = "ACDFFKKIIKKLLMMNNPPQQQRRRRIIIIRRR"


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
    Test that the AptaNetPSeAAC transform method raises an error for protein sequences
    of length smaller or equal to lambda_val.
    """
    p = AptaNetPSeAAC(lambda_val=lambda_val)
    with pytest.raises(ValueError, match="Protein sequence is too short"):
        p.transform(seq)


@pytest.mark.parametrize(
    "seq,expected_vector",
    [
        (
            vector,
            solution,
        )
    ],
)
def test_pseaac_vectorization(seq, expected_vector):
    """
    Test that the AptaNetPSeAAC vectorization produces the expected feature vector.

    Parameters
    ----------
    seq : str
        Protein sequence to transform.
    expected_vector : list of float
        Expected AptaNetPSeAAC feature vector.

    Asserts
    -------
    The produced vector matches the expected vector in length and
    values (within tolerance).
    """
    p = AptaNetPSeAAC()
    pv = p.transform(seq)

    assert len(pv) == len(expected_vector), (
        f"Vector length mismatch: {len(pv)} != {len(expected_vector)}"
    )
    mismatches = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(pv, expected_vector, strict=False))
        if not np.isclose(a, b, atol=1e-3)
    ]
    assert not mismatches, f"Vector values mismatch at indices: {mismatches}"
