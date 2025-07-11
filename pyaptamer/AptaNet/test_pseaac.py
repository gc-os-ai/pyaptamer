import numpy as np
import pytest
from _props import aa_props, solution
from pseaac import PSeAAC


def _normalize_properties(property_dicts):
    """
    Takes multiple amino acid property dictionaries and returns their
    normalized versions.
    Normalization: (value - mean) / std deviation
    Returns a list of normalized dictionaries in the same order.
    """
    normalized = []
    for prop in property_dicts:
        values = list(prop.values())
        mean_val = sum(values) / len(values)
        std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
        normalized.append(
            {aa: round((val - mean_val) / std_val, 3) for aa, val in prop.items()}
        )
    return normalized


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
def test_pseaac_vectorization(seq, expected_vector):
    """
    Test that the PSeAAC vectorization produces the expected feature vector.

    Parameters
    ----------
    seq : str
        Protein sequence to transform.
    expected_vector : list of float
        Expected PSeAAC feature vector.

    Asserts
    -------
    The produced vector matches the expected vector in
    length and values (within tolerance).
    """
    p = PSeAAC()
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
