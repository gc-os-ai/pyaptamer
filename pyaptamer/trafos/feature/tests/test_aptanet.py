"""Tests for AptaNet feature extraction transformations."""

__author__ = ["NandiniDhanrale"]

import numpy as np
import pandas as pd

from pyaptamer.trafos.feature import (
    AptaNetKmerTransformer,
    AptaNetPairTransformer,
    AptaNetPSeAACTransformer,
)
from pyaptamer.utils._aptanet_utils import generate_kmer_vecs, pairs_to_features


def test_aptanet_kmer_transformer_matches_legacy_function():
    """Check the k-mer transformer matches the legacy utility function."""
    X = pd.DataFrame({"aptamer": ["ATG"]})

    Xt = AptaNetKmerTransformer(k=2).fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == (1, 20)
    assert np.allclose(Xt.to_numpy()[0], generate_kmer_vecs("ATG", k=2))


def test_aptanet_kmer_transformer_normalizes_rna_u_to_dna_t():
    """Check RNA U bases are mapped to the AptaNet DNA k-mer convention."""
    X_dna = pd.DataFrame({"aptamer": ["ATG"]})
    X_rna = pd.DataFrame({"aptamer": ["AUG"]})

    vec_dna = AptaNetKmerTransformer(k=2).fit_transform(X_dna)
    vec_rna = AptaNetKmerTransformer(k=2).fit_transform(X_rna)

    assert np.array_equal(vec_rna.to_numpy(), vec_dna.to_numpy())


def test_aptanet_pseaac_transformer_returns_dataframe():
    """Check the PSeAAC transformer returns numeric feature columns."""
    protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    X = pd.DataFrame({"protein": [protein]})

    Xt = AptaNetPSeAACTransformer().fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == 1
    assert Xt.shape[1] == 350
    assert Xt.index.equals(X.index)


def test_aptanet_pair_transformer_matches_pairs_to_features():
    """Check pair transformer matches the legacy pair feature utility."""
    protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    X = pd.DataFrame({"aptamer": ["ATG"], "protein": [protein]})

    Xt = AptaNetPairTransformer(k=2).fit_transform(X)
    legacy = pairs_to_features([("ATG", protein)], k=2)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == legacy.shape
    assert np.allclose(Xt.to_numpy().astype(np.float32), legacy)


def test_aptanet_pair_transformer_accepts_tuple_pairs():
    """Check pair transformer accepts the tuple input used by AptaNetPipeline."""
    protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    pairs = [("ATG", protein)]

    Xt = AptaNetPairTransformer(k=2).fit_transform(pairs)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == 1
