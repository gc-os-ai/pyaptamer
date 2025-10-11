__author__ = "satvshr"

import numpy as np
import pytest

from pyaptamer.pseaac import AptaNetPSeAAC, PSeAAC
from pyaptamer.pseaac._props import aa_props
from pyaptamer.pseaac.tests._props import solution

vector = "ACDFFKKIIKKLLMMNNPPQQQRRRRIIIIRRR"


def test_normalized_values():
    """
    Test that normalized property matrix matches expected normalized values.

    This test targets the common `aa_props` normalization and therefore uses
    the package-level `aa_props` implementation.
    """
    original_df = aa_props(type="pandas", normalize=False)
    normalized_df = aa_props(type="pandas", normalize=True)

    manual_norm = (original_df - original_df.mean()) / original_df.std(ddof=0)
    manual_norm = manual_norm.round(3)

    for aa in original_df.index:
        for prop in original_df.columns:
            assert normalized_df.loc[aa, prop] == manual_norm.loc[aa, prop], (
                f"Mismatch for {aa}, {prop}: "
                f"{normalized_df.loc[aa, prop]} != {manual_norm.loc[aa, prop]}"
            )


@pytest.mark.parametrize("PCLASS", [PSeAAC, AptaNetPSeAAC])
@pytest.mark.parametrize(
    "seq,lambda_val",
    [
        ("ACDEFGHIK", 10),  # length 9, lambda_val 10
        ("ACDAA", 5),  # length 5, lambda_val 5
        ("A", 2),  # length 1, lambda_val 2
    ],
)
def test_pseaac_transform_sequence_too_short(PCLASS, seq, lambda_val):
    """
    Both PSeAAC implementations must raise for sequences shorter than or equal
    to `lambda_val`.
    """
    p = PCLASS(lambda_val=lambda_val)
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
    Test that the AptaNet-specific PSeAAC vectorization produces the expected
    feature vector. This compares against the provided `solution` which matches
    the AptaNet implementation.
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


@pytest.mark.parametrize(
    "seq,prop_indices,group_props,custom_groups,expected_len",
    [
        # Test case 1: default props, group of 3 (should result in 7 groups * 50 = 350)
        (vector, None, None, None, 350),
        # Test case 2: select only 6 props, group into 2 (3 groups * 50 = 150)
        (vector, [0, 1, 2, 3, 4, 5], 2, None, 150),
        # Test case 3: custom grouping of 4 groups(4 groups * 50 = 200)
        (vector, None, None, [[0, 1], [2, 3], [4, 5], [6, 7]], 200),
    ],
)
def test_pseaac_configurations(
    seq,
    prop_indices,
    group_props,
    custom_groups,
    expected_len,
):
    """
    Test different PSeAAC configurations with various property groupings.

    Parameters
    ----------
    seq : str
        Protein sequence to transform.
    prop_indices : list of int or None
        Property indices to use (0-based).
    group_props : int or None
        Grouping size for automatic chunking.
    custom_groups : list of list of int or None
        Custom property groups.
    expected_len : int
        Expected length of resulting feature vector.

    Asserts
    -------
    Output feature vector has the expected length.
    """
    pse = PSeAAC(
        prop_indices=prop_indices, group_props=group_props, custom_groups=custom_groups
    )
    vec = pse.transform(seq)

    assert len(vec) == expected_len, (
        f"Expected vector length {expected_len}, but got {len(vec)}"
    )
