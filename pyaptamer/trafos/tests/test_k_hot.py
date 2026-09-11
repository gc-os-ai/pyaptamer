"""Tests for the SequenceKHotEncoder transform."""

__author__ = ["aditi-dsi"]

import pandas as pd
import pytest
import torch

from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.trafos.encode import SequenceKHotEncoder
from pyaptamer.trafos.transform import PrimerTrimmer

SEQUENCE = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
START_PRIMER = "TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG"
END_PRIMER = "TATGTGCGCATACATGGATCCTC"
TRIMMED_LENGTH = 40


def test_sequence_k_hot_encoder_rejects_invalid_handle_unknown():
    """An unsupported handle_unknown value raises a ValueError."""
    X = pd.DataFrame({"seq": ["ACGT"]})
    with pytest.raises(ValueError, match="handle_unknown must be one of"):
        SequenceKHotEncoder(handle_unknown="xyz").fit_transform(X)


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame({"seq": [1234, 5678]}),
        MoleculeLoader(data={"seq": [["ACGT", "GCTA"]]}),
    ],
    ids=["numeric", "bag_tiling"],
)
def test_sequence_k_hot_encoder_rejects_non_string_column(X):
    """Non-string cells raise TypeError, including bag-tiled multi-sequence cells."""
    with pytest.raises(TypeError, match="one str sequence per row"):
        SequenceKHotEncoder().fit_transform(X)


def test_sequence_k_hot_encoder_raises_on_variable_sequence_lengths():
    """Sequences of unequal length raise a ValueError."""
    X = MoleculeLoader(data={"seq": ["ACGT", "ACGTACGT"]})
    with pytest.raises(ValueError, match="same fixed length"):
        SequenceKHotEncoder().fit_transform(X)


def test_sequence_k_hot_encoder_accepts_moleculeloader():
    """A MoleculeLoader of sequences encodes to a tensor of the expected shape."""
    X = MoleculeLoader(data={"seq": [SEQUENCE, SEQUENCE.lower()]})

    encoder = SequenceKHotEncoder()
    Xt = encoder.fit_transform(X)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (2, 4, len(SEQUENCE))
    assert Xt.dtype == torch.float32
    assert torch.equal(Xt[0], Xt[1])


def test_sequence_k_hot_encoder_custom_columns():
    """Column names are configurable, not hardcoded."""
    X = MoleculeLoader(data={"custom_seq": [SEQUENCE] * 2})
    encoder = SequenceKHotEncoder()
    Xt = encoder.fit_transform(X)

    assert Xt.shape == (2, 4, len(SEQUENCE))


def test_sequence_k_hot_encoder_custom_vocab():
    """A custom vocab sets the class count, and encodes and decodes correctly."""
    encoder = SequenceKHotEncoder(
        vocab={"A": 0, "C": 1, "G": 2, "U": 3, "N": 4},
        inverse_vocab={0: "A", 1: "C", 2: "G", 3: "U", 4: "N"},
    )
    Xt = encoder.fit_transform(pd.DataFrame({"seq": ["ACGN"]}))

    assert Xt.shape == (1, 5, 4)
    assert encoder.inverse_transform(Xt)["sequence"].iloc[0] == "ACGN"


def test_sequence_k_hot_encoder_empty_input():
    """An empty frame encodes to an empty batch, doesn't throw an error."""
    X = pd.DataFrame({"seq": pd.Series([], dtype=object)})
    Xt = SequenceKHotEncoder().fit_transform(X)
    assert Xt.shape == (0, 4, 0)


def test_sequence_k_hot_encoder_handle_unknown_raise_on_unsupported_character():
    """handle_unknown='raise' (default) rejects a sequence containing 'N'."""
    X = MoleculeLoader(data={"seq": ["ACGTN"]})
    with pytest.raises(ValueError, match="unsupported character"):
        SequenceKHotEncoder().fit_transform(X)


@pytest.mark.parametrize("missing_value", [None, float("nan")])
def test_sequence_k_hot_encoder_handle_unknown_raise_on_missing_value(missing_value):
    """handle_unknown='raise' (default) rejects a row containing None/NaN value."""
    X = pd.DataFrame({"seq": ["ACGT", missing_value]})
    with pytest.raises(ValueError, match="None/NaN value"):
        SequenceKHotEncoder().fit_transform(X)


def test_sequence_k_hot_encoder_handle_unknown_drop_unsupported_character():
    """handle_unknown='drop' skips only the sequence containing 'N'."""
    X = MoleculeLoader(data={"sequence": ["ACGT", "ACGN"]})
    encoder = SequenceKHotEncoder(handle_unknown="drop")
    Xt = encoder.fit_transform(X)

    assert Xt.shape == (1, 4, 4)
    assert encoder.inverse_transform(Xt)["sequence"].tolist() == ["ACGT"]


@pytest.mark.parametrize("missing_value", [None, float("nan")])
def test_sequence_k_hot_encoder_handle_unknown_drop_missing_value(missing_value):
    """handle_unknown='drop' skips only the row containing None/NaN value."""
    X = pd.DataFrame({"seq": ["ACGT", missing_value]})
    encoder = SequenceKHotEncoder(handle_unknown="drop")
    Xt = encoder.fit_transform(X)

    assert Xt.shape == (1, 4, 4)
    assert encoder.inverse_transform(Xt)["sequence"].tolist() == ["ACGT"]


def test_sequence_k_hot_encoder_warns_when_all_sequences_dropped():
    """Dropping every sequence warns rather than failing silently."""
    X = pd.DataFrame({"seq": ["ACGN"]})
    encoder = SequenceKHotEncoder(handle_unknown="drop")

    with pytest.warns(UserWarning, match="dropped all"):
        Xt = encoder.fit_transform(X)

    assert Xt.shape == (0, 4, 4)


def test_sequence_k_hot_encoder_inverse_transform():
    """Encoding then decoding returns the original sequences."""
    X = pd.DataFrame({"seq": [SEQUENCE]})
    encoder = SequenceKHotEncoder()

    decoded_df = encoder.inverse_transform(encoder.fit_transform(X))

    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df["sequence"].iloc[0] == SEQUENCE


def test_sequence_k_hot_encoder_inverse_transform_unknown_token():
    """An out-of-bounds index safely decodes to the fallback token 'X'."""
    encoder = SequenceKHotEncoder()

    vocab = {"A": 0, "T": 1, "G": 2, "C": 3}
    real_indices = [vocab[char] for char in SEQUENCE]

    row_with_unknown = real_indices.copy()
    row_with_unknown[10] = 99

    dummy_indices = torch.tensor([row_with_unknown])
    decoded_df = encoder.inverse_transform(dummy_indices)

    expected_mutated = SEQUENCE[:10] + "X" + SEQUENCE[11:]
    assert decoded_df["sequence"].iloc[0] == expected_mutated


def test_sequence_k_hot_encoder_accepts_primer_trimmer_output():
    """Encoder accepts PrimerTrimmer's DataFrame output directly."""
    loader = load_sample_fastq()
    trimmed = PrimerTrimmer(START_PRIMER, END_PRIMER, TRIMMED_LENGTH).fit_transform(
        loader
    )

    assert len(trimmed) > 0

    encoder = SequenceKHotEncoder()
    Xt = encoder.fit_transform(trimmed)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (len(trimmed), 4, TRIMMED_LENGTH)
    assert Xt.dtype == torch.float32

    decoded = encoder.inverse_transform(Xt)
    assert decoded["sequence"].tolist() == trimmed["sequence"].tolist()
