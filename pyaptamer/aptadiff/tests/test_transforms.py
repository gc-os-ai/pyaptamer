"""Tests for the AptamerOneHotEncoder transform."""

__author__ = ["aditi-dsi"]

import pandas as pd
import pytest
import torch

from pyaptamer.aptadiff import AptamerOneHotEncoder
from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.trafos.transform import PrimerTrimmer

APTAMER = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
START_PRIMER = "TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG"
END_PRIMER = "TATGTGCGCATACATGGATCCTC"
TRIMMED_LENGTH = 40


def test_aptamer_one_hot_encoder_accepts_moleculeloader():
    """
    A MoleculeLoader of aptamers is correctly encoded into a tensor
    of shape (batch_size, num_classes, seq_len).
    """
    X = MoleculeLoader(data={"aptamer": [APTAMER, APTAMER.lower()]})

    encoder = AptamerOneHotEncoder()
    Xt = encoder.fit_transform(X)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (2, 4, len(APTAMER))
    assert Xt.dtype == torch.float32
    assert torch.equal(Xt[0], Xt[1])


def test_aptamer_one_hot_encoder_custom_columns():
    """Column names are configurable, not hardcoded to aptamer."""
    X = MoleculeLoader(data={"custom_seq": [APTAMER] * 2})
    encoder = AptamerOneHotEncoder(aptamer_col="custom_seq")
    Xt = encoder.fit_transform(X)

    assert len(Xt) == 2
    assert Xt.shape == (2, 4, len(APTAMER))


def test_aptamer_one_hot_encoder_inverse_transform():
    """A numeric tensor decodes back to string sequences"""
    encoder = AptamerOneHotEncoder()

    vocab = {"A": 0, "T": 1, "G": 2, "C": 3}
    real_indices = [vocab[char] for char in APTAMER]

    dummy_indices = torch.tensor([real_indices])
    decoded_df = encoder.inverse_transform(dummy_indices)

    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df["aptamer"].iloc[0] == APTAMER


def test_aptamer_one_hot_encoder_inverse_transform_unknown_token():
    """An out-of-bounds index safely decodes to the fallback token 'X'."""
    encoder = AptamerOneHotEncoder()

    vocab = {"A": 0, "T": 1, "G": 2, "C": 3}
    real_indices = [vocab[char] for char in APTAMER]

    row_with_unknown = real_indices.copy()
    row_with_unknown[10] = 99

    dummy_indices = torch.tensor([row_with_unknown])
    decoded_df = encoder.inverse_transform(dummy_indices)

    expected_mutated = APTAMER[:10] + "X" + APTAMER[11:]
    assert decoded_df["aptamer"].iloc[0] == expected_mutated


def test_aptamer_one_hot_encoder_accepts_primer_trimmer_output():
    """Encoder accepts PrimerTrimmer's dataframe output directly"""
    loader = load_sample_fastq()
    trimmed = PrimerTrimmer(START_PRIMER, END_PRIMER, TRIMMED_LENGTH).fit_transform(
        loader
    )

    encoder = AptamerOneHotEncoder(aptamer_col="sequence")
    Xt = encoder.fit_transform(trimmed)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (len(trimmed), 4, TRIMMED_LENGTH)
    assert Xt.dtype == torch.float32

    decoded = encoder.inverse_transform(Xt)
    assert decoded["sequence"].tolist() == trimmed["sequence"].tolist()


def test_aptamer_one_hot_encoder_handle_unknown_raise_on_unsupported_character():
    """handle_unknown='raise' (default) rejects a sequence containing 'N'."""
    X = MoleculeLoader(data={"aptamer": ["ACGTN"]})
    with pytest.raises(ValueError, match="unsupported character"):
        AptamerOneHotEncoder().fit_transform(X)


def test_aptamer_one_hot_encoder_handle_unknown_raise_on_missing_value():
    """handle_unknown='raise' (default) rejects a missing sequence value."""
    X = pd.DataFrame({"aptamer": ["ACGT", None]})
    with pytest.raises(ValueError, match="missing value"):
        AptamerOneHotEncoder().fit_transform(X)


def test_aptamer_one_hot_encoder_handle_unknown_drop_unsupported_character():
    """handle_unknown='drop' skips only the row containing 'N'."""
    X = MoleculeLoader(data={"aptamer": ["ACGT", "ACGN"]})
    Xt = AptamerOneHotEncoder(handle_unknown="drop").fit_transform(X)
    assert Xt.shape == (1, 4, 4)


def test_aptamer_one_hot_encoder_handle_unknown_drop_missing_value():
    """handle_unknown='drop' skips only the row with a missing value."""
    X = pd.DataFrame({"aptamer": ["ACGT", None]})
    Xt = AptamerOneHotEncoder(handle_unknown="drop").fit_transform(X)
    assert Xt.shape == (1, 4, 4)


def test_aptamer_one_hot_encoder_raises_on_variable_sequence_lengths():
    """Sequences of unequal length raise a ValueError pointing to PrimerTrimmer."""
    X = MoleculeLoader(data={"aptamer": ["ACGT", "ACGTACGT"]})
    with pytest.raises(ValueError, match="same fixed length"):
        AptamerOneHotEncoder().fit_transform(X)
