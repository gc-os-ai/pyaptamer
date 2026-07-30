"""Test suite for AptaMCTS iCTF encoders."""

__author__ = ["aditi-dsi"]

import numpy as np
import pytest

from pyaptamer.utils._aptamcts_utils import protein_to_ictf, rna_to_ictf


@pytest.mark.parametrize(
    "sequence, k, expected_shape",
    [("ACGU", 1, 4), ("ACGU", 2, 20), ("ACGU", 4, 340)],
)
def test_rna_to_ictf_shapes(sequence, k, expected_shape):
    """Check the shape of the generated iCTF RNA features."""
    assert rna_to_ictf(sequence, k=k).shape == (expected_shape,)


def test_rna_to_ictf_correctness():
    """Check values, DNA-to-RNA conversion, case-insensitivity, and empty strings."""
    expected_k1 = [0.25, 0.25, 0.25, 0.25]

    assert np.allclose(rna_to_ictf("ACGU", k=1), expected_k1)
    assert np.allclose(rna_to_ictf("acGt", k=1), expected_k1)
    assert np.array_equal(rna_to_ictf("", k=2), np.zeros(20))


@pytest.mark.parametrize(
    "sequence, k, expected_shape",
    [("AVIL", 1, 7), ("AVIL", 2, 56), ("AVIL", 3, 399)],
)
def test_protein_to_ictf_shapes(sequence, k, expected_shape):
    """Check the shape of the generated iCTF protein features."""
    assert protein_to_ictf(sequence, k=k).shape == (expected_shape,)


def test_protein_to_ictf_correctness():
    """Check grouping, case-insensitivity, unknown characters, and empty strings."""
    expected_k1 = np.ones(7) / 7

    assert np.allclose(protein_to_ictf("AIYHRDC", k=1), expected_k1)
    assert np.allclose(protein_to_ictf("aiyhrdc", k=1), expected_k1)
    assert np.allclose(protein_to_ictf("AZ", k=1), [0.5, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(protein_to_ictf("", k=2), np.zeros(56))
