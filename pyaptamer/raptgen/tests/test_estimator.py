"""Tests for RaptGenModel and RaptGenGenerator"""

__author__ = ["siddharth7113"]

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.pipeline import Pipeline

from pyaptamer.data import MoleculeLoader
from pyaptamer.raptgen import RaptGenGenerator, RaptGenModel
from pyaptamer.raptgen.layers._sampler import ProfileHMMSampler
from pyaptamer.transformations.encode import SequenceIndexEncoder

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
        "max_epochs": 2,
        "patience": 1,
        "random_state": 0,
    }
    params.update(kwargs)
    return RaptGenModel(**params)


def _encoded():
    return SequenceIndexEncoder().fit_transform(pd.DataFrame({"aptamer": SEQS}))


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


def test_fit_records_epochs():
    """n_epochs_ holds the number of epochs the Trainer ran."""
    model = _model(max_epochs=3, patience=None).fit(_encoded())
    assert model.n_epochs_ == 3


def test_fit_rejects_single_sequence():
    """fit needs at least two sequences for the validation split."""
    with pytest.raises(ValueError, match="at least 2"):
        _model().fit(_encoded()[:1])


def test_generator_roundtrip():
    """RaptGenGenerator takes a MoleculeLoader and returns latents and strings."""
    X = MoleculeLoader(data={"aptamer": SEQS})
    gen = RaptGenGenerator(estimator=_model()).fit(X)
    Z = gen.transform(X)
    assert Z.shape == (len(SEQS), 10)
    assert all(set(seq) <= set("ATGC") for seq in gen.inverse_transform(Z[:2]))
    assert gen.estimator_.random_state == 0


def test_generator_clones_estimator():
    """fit stores a fitted clone and leaves the passed estimator untouched."""
    model = _model()
    gen = RaptGenGenerator(estimator=model).fit(pd.DataFrame({"aptamer": SEQS}))
    assert gen.estimator_ is not model
    assert not hasattr(model, "model_")
    assert isinstance(gen.estimator_, RaptGenModel)


def test_generator_rejects_missing_column():
    """A frame without the aptamer column raises a ValueError naming the columns."""
    with pytest.raises(ValueError, match="aptamer_col"):
        RaptGenGenerator().fit(pd.DataFrame({"seq": SEQS}))


def test_generator_rejects_mixed_lengths():
    """Sequences of different length raise a ValueError."""
    with pytest.raises(ValueError, match="same fixed length"):
        RaptGenGenerator().fit(pd.DataFrame({"aptamer": ["ATGC", "ATG"]}))


def test_generator_as_last_step_of_sklearn_pipeline():
    """A scikit-learn Pipeline ending in the generator gives one latent per row."""
    X = pd.DataFrame({"aptamer": SEQS})
    pipe = Pipeline([("gen", RaptGenGenerator(estimator=_model()))]).fit(X)
    assert pipe.transform(X).shape == (len(SEQS), 10)


def test_transform_is_encoder_mean():
    """transform returns h2mu(encoder(X)), the latent mean, not a sample."""
    X = _encoded()
    model = _model().fit(X)
    device = next(model.model_.parameters()).device
    X_idx = torch.tensor(X.to_numpy(), device=device)
    with torch.no_grad():
        expected = model.model_.h2mu(model.model_.encoder(X_idx))
    np.testing.assert_allclose(model.transform(X), expected.cpu().numpy(), atol=1e-6)


def test_sampled_sequence_is_cleaned(monkeypatch):
    """Lowercase inserts are upper-cased and deletion markers removed."""
    model = _model().fit(_encoded())
    monkeypatch.setattr(
        ProfileHMMSampler, "sample", lambda self, sequence_only: "taccG_TCGtaaAA_"
    )
    Z = model.transform(_encoded()[:1])
    assert model.inverse_transform(Z, most_likely=False) == ["TACCGTCGTAAAA"]


def test_decode_most_probable_fills_wildcards_with_best_scoring(monkeypatch):
    """Deletions are dropped and each N takes the best-scoring nucleotide."""
    model = _model().fit(_encoded())
    monkeypatch.setattr(
        ProfileHMMSampler, "most_probable", lambda self, sequence_only: "A_NCN"
    )
    monkeypatch.setattr(
        ProfileHMMSampler, "calc_seq_proba", lambda self, seq: seq.count("G")
    )
    Z = model.transform(_encoded()[:1])
    assert model.inverse_transform(Z, most_likely=True) == ["AGCG"]
