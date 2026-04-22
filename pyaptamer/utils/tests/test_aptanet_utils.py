"""Tests for AptaNet utility functions."""

import pytest

from pyaptamer.utils._aptanet_utils import generate_kmer_vecs


def test_generate_kmer_vecs_raises_for_rna_sequence():
    """RNA bases should fail clearly because AptaNet expects DNA input."""
    with pytest.raises(ValueError, match="DNA bases only"):
        generate_kmer_vecs("AUG", k=2)
