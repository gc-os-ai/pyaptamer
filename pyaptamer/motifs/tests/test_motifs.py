"""Tests for motif discovery module."""

import numpy as np
import pandas as pd
import pytest

from pyaptamer.motifs import EnrichmentTracker, MotifFinder


class TestMotifFinder:
    """Tests for MotifFinder."""

    @pytest.fixture
    def sequences(self):
        return [
            "ACGTACGTACGT",
            "TACGTACGTACG",
            "GACGTACGTACG",
            "AACGTACGTAAA",
            "CCACGTACGTCC",
        ]

    def test_fit_returns_self(self, sequences):
        finder = MotifFinder(k=4, top_n=3)
        result = finder.fit(sequences)
        assert result is finder

    def test_fit_populates_attributes(self, sequences):
        finder = MotifFinder(k=4, top_n=3)
        finder.fit(sequences)
        assert finder.n_sequences_ == 5
        assert len(finder.kmer_counts_) > 0
        assert isinstance(finder.kmer_scores_, pd.DataFrame)
        assert len(finder.motifs_) <= 3

    def test_motif_has_expected_keys(self, sequences):
        finder = MotifFinder(k=4, top_n=1)
        finder.fit(sequences)
        motif = finder.motifs_[0]
        assert "consensus" in motif
        assert "pwm" in motif
        assert "info_content" in motif
        assert "n_occurrences" in motif
        assert len(motif["consensus"]) == 4

    def test_transform_returns_dataframe(self, sequences):
        finder = MotifFinder(k=4, top_n=2)
        finder.fit(sequences)
        result = finder.transform(sequences)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_fit_transform(self, sequences):
        finder = MotifFinder(k=4, top_n=2)
        result = finder.fit_transform(sequences)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_get_pwm(self, sequences):
        finder = MotifFinder(k=4, top_n=1)
        finder.fit(sequences)
        pwm = finder.get_pwm(0)
        assert isinstance(pwm, pd.DataFrame)
        assert pwm.shape == (4, 4)
        np.testing.assert_allclose(pwm.sum(axis=1).values, 1.0)

    def test_dataframe_input(self, sequences):
        df = pd.DataFrame({"seq": sequences})
        finder = MotifFinder(k=4, top_n=2)
        finder.fit(df)
        assert finder.n_sequences_ == 5

    def test_rna_alphabet(self):
        sequences = ["ACGUACGU", "UACGUACG", "GACGUACG"]
        finder = MotifFinder(k=4, top_n=2, alphabet="RNA")
        finder.fit(sequences)
        assert len(finder.motifs_) > 0

    def test_invalid_alphabet(self, sequences):
        finder = MotifFinder(k=4, alphabet="PROTEIN")
        with pytest.raises(ValueError, match="Unknown alphabet"):
            finder.fit(sequences)


class TestEnrichmentTracker:
    """Tests for EnrichmentTracker."""

    @pytest.fixture
    def rounds(self):
        return [
            ["ACGT", "ACGT", "TGCA", "AAAA", "CCCC", "GGGG"],
            ["ACGT", "ACGT", "ACGT", "TGCA", "AAAA", "ACGT"],
        ]

    def test_add_round_returns_self(self, rounds):
        tracker = EnrichmentTracker(min_count=1)
        result = tracker.add_round(rounds[0])
        assert result is tracker

    def test_compute_requires_two_rounds(self):
        tracker = EnrichmentTracker()
        tracker.add_round(["ACGT"])
        with pytest.raises(ValueError, match="At least 2 rounds"):
            tracker.compute()

    def test_compute_returns_dataframe(self, rounds):
        tracker = EnrichmentTracker(min_count=1)
        tracker.add_round(rounds[0])
        tracker.add_round(rounds[1])
        result = tracker.compute()
        assert isinstance(result, pd.DataFrame)
        assert "freq_round_0" in result.columns
        assert "freq_round_1" in result.columns
        assert "fold_enrichment_1" in result.columns

    def test_enriched_sequence_ranks_higher(self, rounds):
        tracker = EnrichmentTracker(min_count=1)
        tracker.add_round(rounds[0])
        tracker.add_round(rounds[1])
        result = tracker.compute()
        assert result.index[0] == "ACGT"

    def test_top_enriched(self, rounds):
        tracker = EnrichmentTracker(min_count=1)
        tracker.add_round(rounds[0])
        tracker.add_round(rounds[1])
        top = tracker.top_enriched(n=2)
        assert len(top) == 2
