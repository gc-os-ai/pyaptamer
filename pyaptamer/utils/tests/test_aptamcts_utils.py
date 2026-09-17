"""Test suite for AptaMCTS utility functions (iCTF placeholder encoder)."""

__author__ = ["agastya"]

import numpy as np
import pytest

from pyaptamer.utils._aptamcts_utils import _normalized_char_counts, pairs_to_features


# ---------------------------------------------------------------------------
# _normalized_char_counts tests
# ---------------------------------------------------------------------------


class TestNormalizedCharCounts:
    """Tests for the _normalized_char_counts helper."""

    def test_empty_sequence(self):
        """Check that empty sequence returns zeros."""
        alphabet = ["A", "C", "G", "U"]
        result = _normalized_char_counts("", alphabet)

        assert result.shape == (4,)
        assert np.all(result == 0.0)

    def test_single_char(self):
        """Check single character sequence produces correct frequencies."""
        alphabet = ["A", "C", "G", "U"]
        result = _normalized_char_counts("A", alphabet)

        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_uniform_distribution(self):
        """Check uniform character distribution."""
        alphabet = ["A", "C", "G", "U"]
        result = _normalized_char_counts("ACGU", alphabet)

        expected = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_unknown_characters(self):
        """Check that unknown characters are ignored."""
        alphabet = ["A", "C", "G", "U"]
        result = _normalized_char_counts("AXCYGZ", alphabet)

        # Only A, C, G count; X, Y, Z ignored
        expected = np.array([1 / 3, 1 / 3, 1 / 3, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_unknown_characters(self):
        """Check sequence with only unknown characters returns zeros."""
        alphabet = ["A", "C", "G", "U"]
        result = _normalized_char_counts("XYZ", alphabet)

        assert np.all(result == 0.0)

    def test_output_dtype(self):
        """Check that output is float32."""
        alphabet = ["A", "C", "G", "U"]
        result = _normalized_char_counts("ACGU", alphabet)

        assert result.dtype == np.float32

    @pytest.mark.parametrize(
        "sequence, expected",
        [
            ("AAAA", [1.0, 0.0, 0.0, 0.0]),
            ("CCCC", [0.0, 1.0, 0.0, 0.0]),
            ("GGGG", [0.0, 0.0, 1.0, 0.0]),
            ("UUUU", [0.0, 0.0, 0.0, 1.0]),
            ("AACC", [0.5, 0.5, 0.0, 0.0]),
            ("GGUU", [0.0, 0.0, 0.5, 0.5]),
        ],
    )
    def test_various_sequences(self, sequence, expected):
        """Check normalized counts for various sequences."""
        alphabet = ["A", "C", "G", "U"]
        result = _normalized_char_counts(sequence, alphabet)

        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# pairs_to_features edge case tests
# ---------------------------------------------------------------------------


class TestPairsToFeaturesEdgeCases:
    """Edge case tests for pairs_to_features."""

    def test_empty_aptamer_sequence(self):
        """Check encoding with empty aptamer sequence."""
        X = pairs_to_features([("", "ACDEF")])

        assert X.shape == (1, 27)
        assert X.dtype == np.float32
        # First 5 columns (aptamer frequencies) should be zeros
        assert np.all(X[0, :5] == 0.0)

    def test_empty_target_sequence(self):
        """Check encoding with empty target sequence."""
        X = pairs_to_features([("ACGU", "")])

        assert X.shape == (1, 27)
        assert X.dtype == np.float32
        # Columns 5-25 (target frequencies) should be zeros
        assert np.all(X[0, 5:25] == 0.0)

    def test_both_empty_sequences(self):
        """Check encoding with both sequences empty."""
        X = pairs_to_features([("", "")])

        assert X.shape == (1, 27)
        # Length features should be 0.0 / max(0, 1) = 0.0
        assert X[0, -1] == pytest.approx(0.0, abs=1e-5)
        assert X[0, -2] == pytest.approx(0.0, abs=1e-5)

    def test_single_nucleotide_aptamer(self):
        """Check encoding with single nucleotide aptamer."""
        X = pairs_to_features([("A", "ACDEF")])

        assert X.shape == (1, 27)
        # A frequency should be 1.0
        assert X[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_single_amino_acid_target(self):
        """Check encoding with single amino acid target."""
        X = pairs_to_features([("ACGU", "A")])

        assert X.shape == (1, 27)
        # First 5 columns should have A=1.0 for aptamer
        assert X[0, 0] == pytest.approx(0.25, abs=1e-5)

    def test_non_standard_nucleotides(self):
        """Check that non-standard nucleotides are handled."""
        X = pairs_to_features([("ACGTX", "ACDEF")])

        assert X.shape == (1, 27)
        # X should be ignored in nucleotide counts

    def test_non_standard_amino_acids(self):
        """Check that non-standard amino acids are handled."""
        X = pairs_to_features([("ACGU", "ACDEFX")])

        assert X.shape == (1, 27)
        # X should be ignored in amino acid counts

    def test_mixed_case_sequences(self):
        """Check that mixed case sequences are handled."""
        X_upper = pairs_to_features([("ACGU", "ACDEF")])
        X_mixed = pairs_to_features([("AcGu", "AcDeF")])

        np.testing.assert_array_almost_equal(X_upper, X_mixed)

    def test_dna_sequences_converted(self):
        """Check that DNA T is converted to RNA U."""
        X_dna = pairs_to_features([("ACGT", "ACDEF")])
        X_rna = pairs_to_features([("ACGU", "ACDEF")])

        np.testing.assert_array_equal(X_dna, X_rna)

    def test_long_sequences(self):
        """Check encoding with long sequences."""
        long_aptamer = "ACGU" * 100  # 400 nucleotides
        long_target = "ACDEFGHIKLMNPQRSTVWY" * 10  # 200 amino acids
        X = pairs_to_features([(long_aptamer, long_target)])

        assert X.shape == (1, 27)
        assert X.dtype == np.float32
        # Features should be normalized regardless of length
        assert np.all(np.isfinite(X))

    def test_feature_values_in_valid_range(self):
        """Check that all feature values are in valid range [0, 1]."""
        X = pairs_to_features([("ACGU", "ACDEFGHIKLMNPQRSTVWY")])

        assert np.all(X >= 0.0)
        assert np.all(X <= 1.0)

    def test_multiple_pairs_consistency(self):
        """Check that multiple pairs produce consistent feature vectors."""
        pairs = [
            ("ACGU", "ACDEF"),
            ("GCUA", "FEDCA"),
        ]
        X = pairs_to_features(pairs)

        assert X.shape == (2, 27)
        # Both should have same feature structure
        assert np.all(np.isfinite(X))

    def test_dataframe_with_empty_rows(self):
        """Check DataFrame input with empty sequences."""
        import pandas as pd

        df = pd.DataFrame({"aptamer": ["", "ACGU"], "protein": ["ACDEF", ""]})
        X = pairs_to_features(df)

        assert X.shape == (2, 27)
        assert X.dtype == np.float32

    def test_feature_vector_sum_constraint(self):
        """Check that aptamer frequency columns sum to 1.0 for non-empty sequences."""
        X = pairs_to_features([("ACGUACGU", "ACDEF")])

        # First 5 columns are aptamer frequencies
        aptamer_sum = X[0, :5].sum()
        assert aptamer_sum == pytest.approx(1.0, abs=1e-5)
