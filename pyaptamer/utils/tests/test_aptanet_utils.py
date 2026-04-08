"""Test suite for the AptaNet utility functions."""

__author__ = ["NandiniDhanrale"]

import numpy as np

from pyaptamer.utils._aptanet_utils import generate_kmer_vecs


def test_generate_kmer_vecs_normalizes_rna_u_to_dna_t():
    """Check RNA U bases are mapped to the AptaNet DNA k-mer convention."""
    vec_dna = generate_kmer_vecs("ATG", k=2)
    vec_rna = generate_kmer_vecs("AUG", k=2)

    assert vec_rna.shape == vec_dna.shape
    assert np.isclose(vec_rna.sum(), 1.0)
    assert np.array_equal(vec_rna, vec_dna)
