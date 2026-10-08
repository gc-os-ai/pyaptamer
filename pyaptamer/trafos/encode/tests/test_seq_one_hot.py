"""Tests for the SequenceOneHotEncoder transform."""

__author__ = ["aditi-dsi"]

import numpy as np
import pandas as pd
import pytest
import torch

from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.trafos.encode import SequenceOneHotEncoder
from pyaptamer.trafos.transform import PrimerTrimmer

SEQUENCE = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
START_PRIMER = "TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG"
END_PRIMER = "TATGTGCGCATACATGGATCCTC"
TRIMMED_LENGTH = 40


def test_sequence_one_hot_encoder_rejects_invalid_handle_unknown():
    """An unsupported handle_unknown value raises a ValueError."""
    X = pd.DataFrame({"seq": ["ACGT"]})
    with pytest.raises(ValueError, match="handle_unknown must be one of"):
        SequenceOneHotEncoder(handle_unknown="xyz").fit_transform(X)


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame({"seq": [1234, 5678]}),
        MoleculeLoader(data={"seq": [["ACGT", "GCTA"]]}),
    ],
    ids=["numeric", "bag_tiling"],
)
def test_sequence_one_hot_encoder_rejects_non_string_column(X):
    """Non-string cells raise TypeError, including bag-tiled multi-sequence cells."""
    with pytest.raises(TypeError, match="one str sequence per row"):
        SequenceOneHotEncoder().fit_transform(X)


def test_sequence_one_hot_encoder_raises_on_variable_sequence_lengths():
    """Sequences of unequal length raise a ValueError."""
    X = MoleculeLoader(data={"seq": ["ACGT", "ACGTACGT"]})
    with pytest.raises(ValueError, match="same fixed length"):
        SequenceOneHotEncoder().fit_transform(X)


def test_sequence_one_hot_encoder_accepts_moleculeloader():
    """A MoleculeLoader of sequences encodes to a frame of the expected shape."""
    X = MoleculeLoader(data={"seq": [SEQUENCE, SEQUENCE.lower()]})

    encoder = SequenceOneHotEncoder()
    Xt = encoder.fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == (2, 4 * len(SEQUENCE))
    assert Xt.to_numpy().dtype == np.float32
    assert Xt.iloc[0].equals(Xt.iloc[1])


def test_sequence_one_hot_encoder_custom_columns():
    """Column names are configurable, not hardcoded."""
    X = MoleculeLoader(data={"custom_seq": [SEQUENCE] * 2})
    encoder = SequenceOneHotEncoder()
    Xt = encoder.fit_transform(X)

    assert Xt.shape == (2, 4 * len(SEQUENCE))


def test_sequence_one_hot_encoder_custom_vocab():
    """A custom vocab sets the class count, and encodes and decodes correctly."""
    encoder = SequenceOneHotEncoder(
        vocab={"A": 0, "C": 1, "G": 2, "U": 3, "N": 4},
        inverse_vocab={0: "A", 1: "C", 2: "G", 3: "U", 4: "N"},
    )
    Xt = encoder.fit_transform(pd.DataFrame({"seq": ["ACGN"]}))

    assert Xt.shape == (1, 20)
    assert encoder.inverse_transform(Xt)["sequence"].iloc[0] == "ACGN"


def test_sequence_one_hot_encoder_empty_input():
    """An empty frame encodes to an empty batch, doesn't throw an error."""
    X = pd.DataFrame({"seq": pd.Series([], dtype=object)})
    Xt = SequenceOneHotEncoder().fit_transform(X)
    assert Xt.shape == (0, 0)


def test_sequence_one_hot_encoder_handle_unknown_raise_on_unsupported_character():
    """handle_unknown='raise' (default) rejects a sequence containing 'N'."""
    X = MoleculeLoader(data={"seq": ["ACGTN"]})
    with pytest.raises(ValueError, match="unsupported character"):
        SequenceOneHotEncoder().fit_transform(X)


@pytest.mark.parametrize("missing_value", [None, float("nan")])
def test_sequence_one_hot_encoder_handle_unknown_raise_on_missing_value(missing_value):
    """handle_unknown='raise' (default) rejects a row containing None/NaN value."""
    X = pd.DataFrame({"seq": ["ACGT", missing_value]})
    with pytest.raises(ValueError, match="None/NaN value"):
        SequenceOneHotEncoder().fit_transform(X)


def test_sequence_one_hot_encoder_handle_unknown_drop_unsupported_character():
    """handle_unknown='drop' skips only the sequence containing 'N'."""
    X = MoleculeLoader(data={"sequence": ["ACGT", "ACGN"]})
    encoder = SequenceOneHotEncoder(handle_unknown="drop")
    Xt = encoder.fit_transform(X)

    assert Xt.shape == (1, 16)
    assert encoder.inverse_transform(Xt)["sequence"].tolist() == ["ACGT"]


@pytest.mark.parametrize("missing_value", [None, float("nan")])
def test_sequence_one_hot_encoder_handle_unknown_drop_missing_value(missing_value):
    """handle_unknown='drop' skips only the row containing None/NaN value."""
    X = pd.DataFrame({"seq": ["ACGT", missing_value]})
    encoder = SequenceOneHotEncoder(handle_unknown="drop")
    Xt = encoder.fit_transform(X)

    assert Xt.shape == (1, 16)
    assert encoder.inverse_transform(Xt)["sequence"].tolist() == ["ACGT"]


def test_sequence_one_hot_encoder_warns_when_all_sequences_dropped():
    """Dropping every sequence warns rather than failing silently."""
    X = pd.DataFrame({"seq": ["ACGN"]})
    encoder = SequenceOneHotEncoder(handle_unknown="drop")

    with pytest.warns(UserWarning, match="dropped all"):
        Xt = encoder.fit_transform(X)

    assert Xt.shape == (0, 16)


def test_sequence_one_hot_encoder_inverse_transform():
    """Encoding then decoding returns the original sequences."""
    X = pd.DataFrame({"seq": [SEQUENCE]})
    encoder = SequenceOneHotEncoder()

    decoded_df = encoder.inverse_transform(encoder.fit_transform(X))

    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df["sequence"].iloc[0] == SEQUENCE


def test_sequence_one_hot_encoder_inverse_transform_unknown_token():
    """An out-of-bounds index safely decodes to the fallback token 'X'."""
    encoder = SequenceOneHotEncoder()

    vocab = {"A": 0, "T": 1, "G": 2, "C": 3}
    real_indices = [vocab[char] for char in SEQUENCE]

    row_with_unknown = real_indices.copy()
    row_with_unknown[10] = 99

    dummy_indices = torch.tensor([row_with_unknown])
    decoded_df = encoder.inverse_transform(dummy_indices)

    expected_mutated = SEQUENCE[:10] + "X" + SEQUENCE[11:]
    assert decoded_df["sequence"].iloc[0] == expected_mutated


def test_sequence_one_hot_encoder_inverse_transform_3d_tensor():
    """A (batch, num_classes, seq_len) tensor, as a model emits, decodes directly."""
    encoder = SequenceOneHotEncoder()

    vocab = {"A": 0, "T": 1, "G": 2, "C": 3}
    indices = torch.tensor([[vocab[char] for char in "ACGTAC"]])
    logits = torch.nn.functional.one_hot(indices, 4).float().permute(0, 2, 1)
    logits = logits + 0.1 * torch.rand_like(logits)

    assert logits.shape == (1, 4, 6)

    decoded_df = encoder.inverse_transform(logits)
    assert decoded_df["sequence"].iloc[0] == "ACGTAC"


def test_sequence_one_hot_encoder_inverse_transform_rejects_bad_frame_width():
    """A frame whose width is not a multiple of the vocab size raises a ValueError."""
    X = pd.DataFrame(np.zeros((1, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="multiple of 4 columns"):
        SequenceOneHotEncoder().inverse_transform(X)


@pytest.mark.parametrize(
    "X_tensor",
    [torch.tensor([0, 1, 2, 3]), torch.zeros(1, 4, 4, 1)],
    ids=["1d", "4d"],
)
def test_sequence_one_hot_encoder_inverse_transform_rejects_bad_tensor_dim(X_tensor):
    """Tensors that are not 2D or 3D raise a ValueError naming the dimension."""
    with pytest.raises(ValueError, match="expects a 2D index tensor or a 3D"):
        SequenceOneHotEncoder().inverse_transform(X_tensor)


def test_sequence_one_hot_encoder_accepts_primer_trimmer_output():
    """Encoder accepts PrimerTrimmer's DataFrame output directly."""
    loader = load_sample_fastq()
    trimmed = PrimerTrimmer(START_PRIMER, END_PRIMER, TRIMMED_LENGTH).fit_transform(
        loader
    )

    assert len(trimmed) > 0

    encoder = SequenceOneHotEncoder()
    Xt = encoder.fit_transform(trimmed)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == (len(trimmed), 4 * TRIMMED_LENGTH)
    assert Xt.to_numpy().dtype == np.float32

    decoded = encoder.inverse_transform(Xt)
    assert decoded["sequence"].tolist() == trimmed["sequence"].tolist()
