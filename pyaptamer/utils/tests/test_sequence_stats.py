"""Tests for sequence statistics utilities."""

import pytest

from pyaptamer.utils._sequence_stats import (
    gc_content,
    nucleotide_composition,
    sequence_summary,
)


class TestGCContent:
    """Tests for gc_content function."""

    def test_balanced_dna(self):
        assert gc_content("ACGT") == 0.5

    def test_all_gc(self):
        assert gc_content("GGCC") == 1.0

    def test_no_gc(self):
        assert gc_content("AATT") == 0.0

    def test_rna_sequence(self):
        assert gc_content("ACGU") == 0.5

    def test_case_insensitive(self):
        assert gc_content("acgt") == 0.5

    def test_empty_sequence(self):
        assert gc_content("") == 0.0

    def test_single_g(self):
        assert gc_content("G") == 1.0

    def test_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            gc_content(123)

    def test_mixed_case(self):
        assert gc_content("AcGt") == 0.5


class TestNucleotideComposition:
    """Tests for nucleotide_composition function."""

    def test_simple_dna(self):
        comp = nucleotide_composition("AACG")
        assert comp["A"]["count"] == 2
        assert comp["A"]["frequency"] == 0.5
        assert comp["C"]["count"] == 1
        assert comp["G"]["count"] == 1

    def test_rna_with_uracil(self):
        comp = nucleotide_composition("AAUU")
        assert comp["U"]["count"] == 2
        assert "T" not in comp

    def test_unknown_chars_grouped(self):
        comp = nucleotide_composition("ACXZ")
        assert comp["other"]["count"] == 2

    def test_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            nucleotide_composition(42)

    def test_case_insensitive(self):
        comp = nucleotide_composition("aacg")
        assert comp["A"]["count"] == 2


class TestSequenceSummary:
    """Tests for sequence_summary function."""

    def test_basic_summary(self):
        df = sequence_summary(["ACGT", "GGCC", "AAUU"])
        assert len(df) == 3
        assert df["gc_content"].tolist() == [0.5, 1.0, 0.0]
        assert df["length"].tolist() == [4, 4, 4]

    def test_columns_present(self):
        df = sequence_summary(["ACGT"])
        expected_cols = {"sequence", "length", "gc_content", "A", "C", "G", "T", "U"}
        assert expected_cols == set(df.columns)

    def test_nucleotide_counts(self):
        df = sequence_summary(["AACG"])
        assert df.iloc[0]["A"] == 2
        assert df.iloc[0]["C"] == 1
        assert df.iloc[0]["G"] == 1
        assert df.iloc[0]["T"] == 0

    def test_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            sequence_summary("ACGT")

    def test_empty_list_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            sequence_summary([])

    def test_variable_lengths(self):
        df = sequence_summary(["AC", "ACGT", "ACGTACGT"])
        assert df["length"].tolist() == [2, 4, 8]
