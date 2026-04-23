"""Pytest coverage for protein word extraction utilities."""

import pytest

from pyaptamer.utils import compute_protein_words
from pyaptamer.utils._protein_words import compute_protein_words as module_compute_protein_words

# Run each test against both import surfaces:
# - public package API (`pyaptamer.utils`)
# - direct module API (`pyaptamer.utils.protein_words`)


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_public_and_module_imports_work(func):
    """The helper should be available from both public and module import paths."""
    result = func(["MKL"], min_k=1, max_k=3, apply_frequency_filter=False)

    assert result == {
        "M": 1,
        "K": 1,
        "L": 1,
        "MK": 1,
        "KL": 1,
        "MKL": 1,
    }


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_overlapping_k3_windows(func):
    """k=3 must use overlapping windows, not chunked windows."""
    result = func(["MKLAVT"], min_k=3, max_k=3, apply_frequency_filter=False)

    assert result == {"MKL": 1, "KLA": 1, "LAV": 1, "AVT": 1}


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_mixed_k_1_to_3_uses_all_overlapping_substrings(func):
    """k=1..3 should include all overlapping substrings up to length 3."""
    result = func(["MKL"], min_k=1, max_k=3, apply_frequency_filter=False)

    assert result == {
        "M": 1,
        "K": 1,
        "L": 1,
        "MK": 1,
        "KL": 1,
        "MKL": 1,
    }


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_frequency_filter_removes_below_average_words_and_renumbers(func):
    """Filtering should discard below-average words and renumber survivors."""
    result = func(["AAAB"], min_k=1, max_k=1, apply_frequency_filter=True)

    assert result == {"A": 1}
    # Filtered output is an id mapping, not raw frequencies.
    assert all(isinstance(value, int) for value in result.values())


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_frequency_filter_with_all_equal_frequencies_returns_empty(func):
    """Strictly above-average filtering should drop words tied at the mean."""
    result = func(["AB"], min_k=1, max_k=1, apply_frequency_filter=True)

    assert result == {}


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_skips_empty_none_and_whitespace_inputs(func):
    """None and blank entries should be ignored without errors."""
    result = func([None, "", "   ", " mk "], min_k=1, max_k=1, apply_frequency_filter=False)

    assert result == {"M": 1, "K": 1}


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_accepts_any_iterable_of_sequences(func):
    """The input can be any iterable, not just a list."""
    sequences = (seq for seq in ["AAA", "AAB"])
    result = func(sequences, min_k=1, max_k=1, apply_frequency_filter=False)

    assert result == {"A": 5, "B": 1}


@pytest.mark.parametrize(
    "min_k,max_k",
    [(0, 3), (2, 1), (-1, 2)],
)
@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_invalid_k_bounds_raise_value_error(func, min_k, max_k):
    """Invalid k bounds should fail fast."""
    with pytest.raises(ValueError, match="Expected 1 <= min_k <= max_k"):
        func(["MKL"], min_k=min_k, max_k=max_k)


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_sequences_shorter_than_k_are_skipped(func):
    """Sequences shorter than k should contribute nothing."""
    result = func(["AA", "A"], min_k=3, max_k=3, apply_frequency_filter=False)

    assert result == {}


@pytest.mark.parametrize("func", [compute_protein_words, module_compute_protein_words])
def test_filtering_matches_shared_mean_rule(func):
    """The filtered output should keep only words with frequency above the mean."""
    result = func(["AABBCC", "AAABBB"], min_k=1, max_k=1, apply_frequency_filter=True)

    # Counts are A=5, B=5, C=2 -> mean = 4, so A and B survive.
    assert result == {"A": 1, "B": 2}
