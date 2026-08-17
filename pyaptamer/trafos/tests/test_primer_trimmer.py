"""Tests for the SELEXTransform transformations."""

__author__ = ["aditi-dsi"]

import pandas as pd
import pytest

from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.trafos.transform import SELEXTransform

START_PRIMER = "TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG"
END_PRIMER = "TATGTGCGCATACATGGATCCTC"
TARGET_LENGTH = 100


@pytest.mark.parametrize("start,end", [(None, "ACGT"), ("ACGT", None), (None, None)])
def test_selex_explicit_requires_both_primers(start, end):
    """The explicit method cannot run without providing both primers."""
    with pytest.raises(ValueError, match="Both primers required"):
        SELEXTransform(start_primer=start, end_primer=end).fit(load_sample_fastq())


def test_selex_rejects_unknown_method():
    """A method outside the accepted ones is rejected."""
    with pytest.raises(ValueError, match="method must be"):
        SELEXTransform(method="heuristics").fit(load_sample_fastq())


def test_selex_explicit_strips_primers():
    """Matching reads lose exactly their primers, leaving the variable region."""
    X = load_sample_fastq()
    sequences = X.to_dataframe().iloc[:, 0]

    transform = SELEXTransform(start_primer=START_PRIMER, end_primer=END_PRIMER)
    Xt = transform.fit_transform(X)

    matched = (
        sequences.str.startswith(START_PRIMER)
        & sequences.str.endswith(END_PRIMER)
        & (sequences.str.len() == transform.target_length_)
    )

    assert matched.any()
    assert Xt.iloc[:, 0][matched].notna().all()
    for original, trimmed in zip(
        sequences[matched], Xt.iloc[:, 0][matched], strict=False
    ):
        assert original == START_PRIMER + trimmed + END_PRIMER


def test_selex_output_aligns_with_input():
    """The output is a single-column frame keeping one row per input read."""
    X = load_sample_fastq()
    sequences = X.to_dataframe().iloc[:, 0]

    Xt = SELEXTransform(start_primer=START_PRIMER, end_primer=END_PRIMER).fit_transform(
        X
    )

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == (len(sequences), 1)
    assert Xt.index.equals(sequences.index)


def test_selex_unmatched_reads_become_na():
    """Reads not carrying the given primers are replaced with NA rather than trimmed."""
    Xt = SELEXTransform(start_primer="ZZZZ", end_primer="ZZZZ").fit_transform(
        load_sample_fastq()
    )

    assert Xt.iloc[:, 0].isna().all()


def test_selex_drops_reads_outside_tolerance():
    """Reads away from the target length are dropped unless tolerance allows them."""
    reads = [START_PRIMER + "A" * 40 + END_PRIMER] * 3
    reads.append(START_PRIMER + "A" * 41 + END_PRIMER)
    X = MoleculeLoader(data={"sequence": reads})

    strict = SELEXTransform(start_primer=START_PRIMER, end_primer=END_PRIMER)
    lenient = SELEXTransform(
        start_primer=START_PRIMER, end_primer=END_PRIMER, tolerance=1
    )

    assert strict.fit_transform(X).iloc[:, 0].notna().sum() == 3
    assert lenient.fit_transform(X).iloc[:, 0].notna().sum() == 4


def test_selex_max_len_center_crops():
    """Given max_len, the variable region is cropped from the centre, as in AptaDiff."""
    X = load_sample_fastq()
    max_len = 5

    full = (
        SELEXTransform(start_primer=START_PRIMER, end_primer=END_PRIMER)
        .fit_transform(X)
        .iloc[:, 0]
        .dropna()
    )
    cropped = (
        SELEXTransform(
            start_primer=START_PRIMER, end_primer=END_PRIMER, max_len=max_len
        )
        .fit_transform(X)
        .iloc[:, 0]
        .dropna()
    )

    assert full.index.equals(cropped.index)
    for original, crop in zip(full, cropped, strict=False):
        offset = len(original) // 2 - max_len // 2
        assert crop == original[offset : offset + max_len]


def test_selex_heuristic_infers_primers():
    """The heuristic recovers the primers and target length of the sample data."""
    transform = SELEXTransform(method="heuristic").fit(load_sample_fastq())

    assert transform.start_primer_ == START_PRIMER
    assert transform.end_primer_ == END_PRIMER
    assert transform.target_length_ == TARGET_LENGTH


def test_selex_heuristic_keeps_supplied_primer():
    """A primer given explicitly is kept as-is, only the missing one is inferred."""
    transform = SELEXTransform(method="heuristic", start_primer="ACGT").fit(
        load_sample_fastq()
    )

    assert transform.start_primer_ == "ACGT"
    assert isinstance(transform.end_primer_, str)


def test_selex_rejects_non_moleculeloader():
    """Only a MoleculeLoader is accepted, a plain DataFrame is rejected."""
    X = pd.DataFrame({"sequence": ["ACGTACGTACGT"]})

    with pytest.raises(TypeError, match="only a MoleculeLoader"):
        SELEXTransform(start_primer="ACGT", end_primer="ACGT").fit(X)
