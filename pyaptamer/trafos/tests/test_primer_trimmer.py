"""Tests for the PrimerTrimmer transformation."""

__author__ = ["aditi-dsi", "siddharth7113"]

import warnings

import pandas as pd
import pytest

from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.trafos.transform import PrimerTrimmer

START_PRIMER = "TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG"
END_PRIMER = "TATGTGCGCATACATGGATCCTC"
VARIABLE_LENGTH = 40


def make_read(region):
    """Build a read with the sample primers around region."""
    return START_PRIMER + region + END_PRIMER


def test_strips_primers():
    """A well formed read loses exactly its two constant regions."""
    region = "ACGT" * 10
    Xt = PrimerTrimmer(START_PRIMER, END_PRIMER, VARIABLE_LENGTH).fit_transform(
        MoleculeLoader(data={"sequence": [make_read(region)]})
    )

    assert Xt["sequence"].tolist() == [region]


def test_output_is_a_single_column_frame():
    """The output is a DataFrame keeping the column name and index of the input."""
    X = load_sample_fastq()
    Xt = PrimerTrimmer(START_PRIMER, END_PRIMER, VARIABLE_LENGTH).fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert list(Xt.columns) == ["sequence"]
    assert Xt.index.isin(X.to_dataframe().index).all()


def test_keeps_only_reads_matching_the_design():
    """Reads are kept only if both primers and the region length match."""
    reads = {
        "good": make_read("ACGT" * 10),
        "short_region": make_read("ACGT" * 9),
        "long_region": make_read("ACGT" * 11),
        "no_start": "GGGG" + "ACGT" * 10 + END_PRIMER,
        "no_end": START_PRIMER + "ACGT" * 10 + "GGGG",
    }
    Xt = PrimerTrimmer(START_PRIMER, END_PRIMER, VARIABLE_LENGTH).fit_transform(
        MoleculeLoader(data={"sequence": list(reads.values())})
    )

    assert Xt["sequence"].tolist() == ["ACGT" * 10]


def test_sample_fastq_yields_fixed_length_regions():
    """Every surviving read of the sample data is exactly one random region."""
    Xt = PrimerTrimmer(START_PRIMER, END_PRIMER, VARIABLE_LENGTH).fit_transform(
        load_sample_fastq()
    )

    assert len(Xt) == 6
    assert Xt["sequence"].str.len().unique().tolist() == [VARIABLE_LENGTH]
    assert not Xt["sequence"].isna().any()


def test_no_matching_read_warns_and_gives_an_empty_frame():
    """Primers matching nothing warn rather than silently returning nothing."""
    with pytest.warns(UserWarning, match="dropped all"):
        Xt = PrimerTrimmer("ZZZZ", "ZZZZ", VARIABLE_LENGTH).fit_transform(
            load_sample_fastq()
        )

    assert Xt.empty
    assert list(Xt.columns) == ["sequence"]


def test_partial_drop_does_not_warn():
    """Dropping some reads is the expected filtering, so it stays quiet."""
    reads = [make_read("ACGT" * 10), make_read("ACGT" * 9)]
    X = MoleculeLoader(data={"sequence": reads})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Xt = PrimerTrimmer(START_PRIMER, END_PRIMER, VARIABLE_LENGTH).fit_transform(X)

    assert len(Xt) == 1


def test_accepts_a_dataframe():
    """A DataFrame is accepted, as the base class contract allows."""
    X = load_sample_fastq().to_dataframe()
    Xt = PrimerTrimmer(START_PRIMER, END_PRIMER, VARIABLE_LENGTH).fit_transform(X)

    assert len(Xt) == 6


def test_missing_cells_are_dropped_not_fatal():
    """A missing read is dropped, as it cannot match the design."""
    X = MoleculeLoader(data={"sequence": [make_read("ACGT" * 10), None]})
    Xt = PrimerTrimmer(START_PRIMER, END_PRIMER, VARIABLE_LENGTH).fit_transform(X)

    assert Xt["sequence"].tolist() == ["ACGT" * 10]


def test_bag_tiling_names_the_fix():
    """Multi-read cells are rejected with a message naming the tiling to use."""
    X = MoleculeLoader(data={"sequence": ["pyaptamer/datasets/data/sample.fastq"]})

    with pytest.raises(TypeError, match='tiling="samples"'):
        PrimerTrimmer(START_PRIMER, END_PRIMER, VARIABLE_LENGTH).fit_transform(X)


@pytest.mark.parametrize("start,end", [("", END_PRIMER), (START_PRIMER, ""), ("", "")])
def test_empty_primers_are_rejected(start, end):
    """Both constant regions are required."""
    with pytest.raises(ValueError, match="must be"):
        PrimerTrimmer(start, end, VARIABLE_LENGTH).fit_transform(load_sample_fastq())


@pytest.mark.parametrize("variable_length", [0, -1])
def test_non_positive_variable_length_is_rejected(variable_length):
    """A random region must have a positive designed length."""
    with pytest.raises(ValueError, match="variable_length must be positive"):
        PrimerTrimmer(START_PRIMER, END_PRIMER, variable_length).fit_transform(
            load_sample_fastq()
        )


def test_overlapping_primers_drop_rather_than_return_empty_strings():
    """Primers that cannot both fit leave no read, rather than empty strings."""
    X = MoleculeLoader(data={"sequence": ["ACGTACG"] * 4})

    with pytest.warns(UserWarning, match="dropped all"):
        Xt = PrimerTrimmer("ACGTA", "GTACG", 2).fit_transform(X)

    assert Xt.empty


def test_output_can_be_fed_back_in():
    """The output carries no missing values, so it chains into another trim."""
    core = "ACGT" * 8
    X = MoleculeLoader(data={"sequence": [make_read("AAA" + core + "TTT")]})

    once = PrimerTrimmer(START_PRIMER, END_PRIMER, len(core) + 6).fit_transform(X)
    twice = PrimerTrimmer("AAA", "TTT", len(core)).fit_transform(
        MoleculeLoader(data=once)
    )

    assert once["sequence"].tolist() == ["AAA" + core + "TTT"]
    assert twice["sequence"].tolist() == [core]


@pytest.mark.parametrize("on_unmatched", ["some", "error"])
def test_invalid_on_unmatched_is_rejected(on_unmatched):
    """on_unmatched only accepts 'drop', 'na', or 'raise'."""
    with pytest.raises(ValueError, match="on_unmatched must be one of"):
        PrimerTrimmer(
            START_PRIMER, END_PRIMER, VARIABLE_LENGTH, on_unmatched=on_unmatched
        ).fit_transform(load_sample_fastq())


def test_on_unmatched_raise_aborts_on_unmatched_reads():
    """raise stops on the first unmatched read."""
    reads = [make_read("ACGT" * 10), make_read("ACGT" * 9)]
    X = MoleculeLoader(data={"sequence": reads})

    with pytest.raises(ValueError, match=r"1 of 2"):
        PrimerTrimmer(
            START_PRIMER, END_PRIMER, VARIABLE_LENGTH, on_unmatched="raise"
        ).fit_transform(X)


def test_on_unmatched_raise_passes_when_all_reads_match():
    """raise is a no-op when every read fits the library design."""
    region = "ACGT" * 10
    X = MoleculeLoader(data={"sequence": [make_read(region)]})

    Xt = PrimerTrimmer(
        START_PRIMER, END_PRIMER, VARIABLE_LENGTH, on_unmatched="raise"
    ).fit_transform(X)

    assert Xt["sequence"].tolist() == [region]


def test_on_unmatched_na_preserves_input_length_and_marks_unmatched_reads():
    """na keeps every row, replacing an unmatched read's sequence with None."""
    region = "ACGT" * 10
    reads = [make_read(region), make_read("ACGT" * 9)]
    X = MoleculeLoader(data={"sequence": reads})

    Xt = PrimerTrimmer(
        START_PRIMER, END_PRIMER, VARIABLE_LENGTH, on_unmatched="na"
    ).fit_transform(X)

    assert Xt.index.equals(X.to_dataframe().index)
    assert Xt["sequence"].isna().tolist() == [False, True]
    assert Xt.loc[Xt["sequence"].notna(), "sequence"].tolist() == [region]


def test_on_unmatched_na_all_unmatched_warns_but_keeps_every_row():
    """na warns when every read fails, but keeps the frame's full shape."""
    X = load_sample_fastq()

    with pytest.warns(UserWarning, match="set all"):
        Xt = PrimerTrimmer(
            "ZZZZ", "ZZZZ", VARIABLE_LENGTH, on_unmatched="na"
        ).fit_transform(X)

    assert len(Xt) == len(X.to_dataframe())
    assert Xt["sequence"].isna().all()


def test_on_unmatched_na_partial_does_not_warn():
    """Marking some reads as unmatched is expected filtering, so it stays quiet."""
    reads = [make_read("ACGT" * 10), make_read("ACGT" * 9)]
    X = MoleculeLoader(data={"sequence": reads})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Xt = PrimerTrimmer(
            START_PRIMER, END_PRIMER, VARIABLE_LENGTH, on_unmatched="na"
        ).fit_transform(X)

    assert len(Xt) == 2
