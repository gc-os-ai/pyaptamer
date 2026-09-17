import numpy as np

from pyaptamer.utils import rna2vec


def test_rna2vec_case_insensitivity():
    """
    Test that rna2vec handles secondary structures in a case-insensitive way.
    Regression test for Issue where lowercase 'ss' inputs returned zero vectors.
    """
    sequences_upper = ["SSHH"]
    sequences_lower = ["sshh"]

    result_upper = rna2vec(sequences_upper, sequence_type="ss", max_sequence_length=10)
    result_lower = rna2vec(sequences_lower, sequence_type="ss", max_sequence_length=10)

    # Both should be identical and non-zero
    assert np.array_equal(result_upper, result_lower)
    assert np.any(result_upper != 0)
