"""Tests for RaptGenModel and RaptgenGenerator"""

__author__ = ["siddharth7113"]

import numpy as np
import pytest
import torch

from pyaptamer.raptgen import RaptgenGenerator, RaptGenModel

SEQS = [
    "ATGCATGCATGC",
    "GGCATTACGTAC",
    "TTGACCGTAGCA",
    "CCATGGTACGAT",
    "AGTCAGTCAGTC",
    "GATTACAGATTA",
    "CGCGATATCGCG",
    "TACGTACGTACG",
]


def _model(**kwargs):
    params = {
        "motif_len": 5,
        "hidden_size": 8,
        "epochs": 2,
        "threshold": 1,
        "random_state": 0,
    }
    params.update(kwargs)
    return RaptGenModel(**params)


def _encoded():
    return RaptgenGenerator()._encode_sequences(SEQS)


def test_fit_transform_shape():
    """transform returns one latent vector of size embed_size per sequence."""
    model = _model(embed_size=6).fit(_encoded())
    Z = model.transform(_encoded())
    assert Z.shape == (len(SEQS), 6)


@pytest.mark.parametrize("most_likely", [True, False])
def test_inverse_transform_returns_nucleotides(most_likely):
    """Both decoding modes return plain A/T/G/C strings, one per latent point."""
    model = _model().fit(_encoded())
    Z = model.transform(_encoded()[:3])
    seqs = model.inverse_transform(Z, most_likely=most_likely)
    assert len(seqs) == 3
    assert all(set(seq) <= set("ATGC") for seq in seqs)


def test_random_state_reproducible():
    """Same random_state gives the same latents and the same sampled sequences."""
    X = _encoded()
    m1 = _model().fit(X)
    m2 = _model().fit(X)
    assert np.allclose(m1.transform(X), m2.transform(X))
    Z = m1.transform(X[:4])
    assert m1.inverse_transform(Z, most_likely=False) == m2.inverse_transform(
        Z, most_likely=False
    )


def test_fit_leaves_global_rng_untouched():
    """fit with random_state does not change the global torch RNG state."""
    torch.manual_seed(123)
    _model().fit(_encoded())
    after_fit = torch.rand(1).item()
    torch.manual_seed(123)
    assert after_fit == torch.rand(1).item()


def test_fit_rejects_single_sequence():
    """fit needs at least two sequences for the validation split."""
    with pytest.raises(ValueError, match="at least 2"):
        _model().fit(_encoded()[:1])


def test_generator_roundtrip():
    """RaptgenGenerator accepts strings and returns strings."""
    gen = RaptgenGenerator(motif_len=5)
    gen.model_ = _model().fit(_encoded())
    Z = gen.transform(SEQS[:2])
    assert Z.shape[0] == 2
    assert all(set(seq) <= set("ATGC") for seq in gen.inverse_transform(Z))


def test_generator_rejects_mixed_lengths():
    """Sequences of different length raise a ValueError."""
    with pytest.raises(ValueError, match="same length"):
        RaptgenGenerator()._encode_sequences(["ATGC", "ATG"])


def test_generator_rejects_invalid_characters():
    """Characters outside A, T, G, C, U raise a ValueError naming them."""
    with pytest.raises(ValueError, match=r"\['N', 'a'\]"):
        RaptgenGenerator()._encode_sequences(["ATGN", "aTGC"])
