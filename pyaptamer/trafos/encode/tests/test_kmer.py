"""Unit tests for KMerEncoder — Issue #696.

Tests verify that the RNA 'U' nucleotide bug is fixed and that
the auto-inferred and explicit alphabet features work correctly.
"""

import numpy as np
import pandas as pd
import pytest

from pyaptamer.trafos.encode._kmer import KMerEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(sequences):
    """Create a single-column DataFrame of sequences."""
    return pd.DataFrame(sequences, columns=["Sequence"])


# ---------------------------------------------------------------------------
# k=1 frequency tests
# ---------------------------------------------------------------------------

class TestKMerEncoderK1:
    """Tests with k=1 (single-character k-mers)."""

    def test_dna_k1_frequencies(self):
        """DNA 'ACGT' with k=1 should yield equal 0.25 frequencies."""
        enc = KMerEncoder(k=1)
        df = _make_df(["ACGT"])
        result = enc.fit_transform(df)

        freqs = result.values[0]
        # Alphabet inferred as ['A', 'C', 'G', 'T'] (sorted)
        expected = [0.25, 0.25, 0.25, 0.25]
        np.testing.assert_allclose(freqs, expected, atol=1e-6)

    def test_rna_k1_frequencies(self):
        """RNA 'ACGU' with k=1 should yield equal 0.25 frequencies (bug fix)."""
        enc = KMerEncoder(k=1)
        df = _make_df(["ACGU"])
        result = enc.fit_transform(df)

        freqs = result.values[0]
        # Alphabet inferred as ['A', 'C', 'G', 'U'] (sorted)
        expected = [0.25, 0.25, 0.25, 0.25]
        np.testing.assert_allclose(freqs, expected, atol=1e-6)

    def test_all_u_sequence(self):
        """'UUUU' with k=1 should give U freq=1.0, not all zeros."""
        enc = KMerEncoder(k=1)
        df = _make_df(["UUUU"])
        result = enc.fit_transform(df)

        freqs = result.values[0]
        # Alphabet inferred as ['U'] only
        assert result.shape[1] == 1
        np.testing.assert_allclose(freqs, [1.0], atol=1e-6)

    def test_rna_u_not_zero(self):
        """Verify U frequency is non-zero for RNA sequences."""
        enc = KMerEncoder(k=1)
        df = _make_df(["ACGU"])
        result = enc.fit_transform(df)

        # All frequencies should be non-zero
        assert np.all(result.values[0] > 0)


# ---------------------------------------------------------------------------
# k=2 tests
# ---------------------------------------------------------------------------

class TestKMerEncoderK2:
    """Tests with k=2 (1-mers and 2-mers)."""

    def test_k2_rna_counts_gu(self):
        """RNA 'ACGU' with k=2 should count 'GU' and 'U' k-mers."""
        enc = KMerEncoder(k=2)
        df = _make_df(["ACGU"])
        result = enc.fit_transform(df)

        # With alphabet ['A','C','G','U'], k=2 generates:
        # 1-mers: A, C, G, U (4)
        # 2-mers: AA, AC, AG, AU, CA, CC, CG, CU, GA, GC, GG, GU, UA, UC, UG, UU (16)
        # Total vocabulary: 20
        assert result.shape[1] == 20

        # All values should be non-negative
        assert np.all(result.values >= 0)

        # Total frequencies should sum to 1.0
        np.testing.assert_allclose(result.values[0].sum(), 1.0, atol=1e-6)

    def test_k2_dna_preserved(self):
        """DNA 'ACGT' with k=2 should produce 7 non-zero k-mers: A,C,G,T,AC,CG,GT."""
        enc = KMerEncoder(k=2)
        df = _make_df(["ACGT"])
        result = enc.fit_transform(df)

        nonzero_count = np.count_nonzero(result.values[0])
        assert nonzero_count == 7


# ---------------------------------------------------------------------------
# Alphabet parameter tests
# ---------------------------------------------------------------------------

class TestKMerEncoderAlphabet:
    """Tests for explicit and inferred alphabet behavior."""

    def test_custom_alphabet_string(self):
        """Explicit alphabet='ACGU' should produce U-containing k-mers."""
        enc = KMerEncoder(k=1, alphabet="ACGU")
        df = _make_df(["ACGT"])  # DNA input, but alphabet has U
        result = enc.fit_transform(df)

        # Should have 4 columns: A, C, G, U
        assert result.shape[1] == 4

        # U frequency should be 0 (not in input), but column exists
        freqs = result.values[0]
        # A=0.25, C=0.25, G=0.25, U=0.0 (T is not in alphabet, silently ignored)
        # Wait — 'T' is NOT in the alphabet 'ACGU', so 'T' is dropped.
        # Only A, C, G are counted → total=3, each freq=1/3
        # U freq=0
        assert freqs[3] == 0.0  # U column exists but is zero

    def test_custom_alphabet_list(self):
        """Explicit alphabet as list should work identically."""
        enc = KMerEncoder(k=1, alphabet=["A", "C", "G", "U"])
        df = _make_df(["ACGU"])
        result = enc.fit_transform(df)

        expected = [0.25, 0.25, 0.25, 0.25]
        np.testing.assert_allclose(result.values[0], expected, atol=1e-6)

    def test_mixed_dna_rna_fit(self):
        """Fitting on mixed DNA+RNA sequences should infer ACGTU alphabet."""
        enc = KMerEncoder(k=1)
        df = _make_df(["ACGT", "ACGU"])
        result = enc.fit_transform(df)

        # Alphabet should be ['A', 'C', 'G', 'T', 'U'] — 5 columns
        assert result.shape[1] == 5

    def test_transform_after_fit_on_different_data(self):
        """Fit on RNA, transform on DNA — U columns should exist but be zero."""
        enc = KMerEncoder(k=1)

        # Fit on RNA data
        df_rna = _make_df(["ACGU"])
        enc.fit(df_rna)

        # Transform DNA data
        df_dna = _make_df(["ACGT"])
        result = enc.transform(df_dna)

        # Alphabet was inferred as ['A', 'C', 'G', 'U']
        assert result.shape[1] == 4

        # 'T' is not in the alphabet, so it's dropped from counting
        # Only A, C, G are counted (3 out of 4 chars)
        freqs = result.values[0]
        # U column (index 3) should be 0
        assert freqs[3] == 0.0

    def test_alphabet_sorted(self):
        """Auto-inferred alphabet should be sorted."""
        enc = KMerEncoder(k=1)
        df = _make_df(["UGCA"])  # Out-of-order characters
        enc.fit(df)

        assert enc.alphabet_ == ["A", "C", "G", "U"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestKMerEncoderEdgeCases:
    """Edge case tests."""

    def test_single_char_sequence(self):
        """Single character sequence should work."""
        enc = KMerEncoder(k=1)
        df = _make_df(["A"])
        result = enc.fit_transform(df)

        assert result.shape == (1, 1)
        np.testing.assert_allclose(result.values[0], [1.0], atol=1e-6)

    def test_multiple_sequences(self):
        """Multiple sequences should produce correct shape."""
        enc = KMerEncoder(k=1)
        df = _make_df(["ACGU", "AAAA", "UUUU"])
        result = enc.fit_transform(df)

        assert result.shape[0] == 3
        # Alphabet inferred from all sequences: ['A', 'C', 'G', 'U']
        assert result.shape[1] == 4

    def test_frequencies_sum_to_one(self):
        """All frequency vectors should sum to 1.0."""
        enc = KMerEncoder(k=2)
        df = _make_df(["ACGUACGU", "AAACCCGGGUUU"])
        result = enc.fit_transform(df)

        for i in range(result.shape[0]):
            np.testing.assert_allclose(
                result.values[i].sum(), 1.0, atol=1e-6,
                err_msg=f"Row {i} frequencies do not sum to 1.0",
            )

    def test_index_preserved(self):
        """Output DataFrame should preserve the input index."""
        df = pd.DataFrame(["ACGU", "UUUU"], columns=["Sequence"], index=[10, 20])
        enc = KMerEncoder(k=1)
        result = enc.fit_transform(df)

        assert list(result.index) == [10, 20]
