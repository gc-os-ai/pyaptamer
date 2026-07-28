"""Tests for the PairsToTokens transform."""

__author__ = ["siddharth7113"]

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from pyaptamer.aptatrans import PairsToTokens
from pyaptamer.data import MoleculeLoader
from pyaptamer.utils import encode_rna, rna2vec
from pyaptamer.utils._base import filter_words

APTAMER = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
PROTEIN = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
PROT_WORDS = {"ACD": 5.0, "EFG": 4.0, "HIK": 3.0, "LMN": 1.0}


def test_output_shape_and_dtype():
    """Output is one row per pair, apta_max_len + prot_max_len integer columns."""
    X = MoleculeLoader(data={"aptamer": [APTAMER] * 3, "protein": [PROTEIN] * 3})

    Xt = PairsToTokens(
        prot_words=PROT_WORDS, apta_max_len=10, prot_max_len=8
    ).fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == (3, 18)
    assert Xt.to_numpy().dtype == np.int64


def test_column_blocks_split_at_apta_max_len():
    """Columns before apta_max_len hold the aptamer block, those after the protein."""
    X = MoleculeLoader(data={"aptamer": [APTAMER] * 3, "protein": [PROTEIN] * 3})

    Xt = PairsToTokens(
        prot_words=PROT_WORDS, apta_max_len=10, prot_max_len=8
    ).fit_transform(X)

    expected_apta = rna2vec([APTAMER] * 3, max_sequence_length=10)
    expected_prot = encode_rna(
        [PROTEIN] * 3,
        words=filter_words(PROT_WORDS),
        max_len=8,
        return_type="numpy",
    )

    np.testing.assert_array_equal(Xt.to_numpy()[:, :10], expected_apta)
    np.testing.assert_array_equal(Xt.to_numpy()[:, 10:], expected_prot)


def test_custom_column_names():
    """Column names are configurable, not hardcoded to aptamer/protein."""
    X = MoleculeLoader(
        data={"aptamer_sequence": [APTAMER] * 2, "target_sequence": [PROTEIN] * 2}
    )

    Xt = PairsToTokens(
        prot_words=PROT_WORDS,
        apta_max_len=10,
        prot_max_len=8,
        aptamer_col="aptamer_sequence",
        protein_col="target_sequence",
    ).fit_transform(X)

    assert Xt.shape == (2, 18)


def test_prot_words_derived_in_fit():
    """prot_words_ is absent before fit and holds the filtered vocabulary after."""
    X = MoleculeLoader(data={"aptamer": [APTAMER], "protein": [PROTEIN]})
    transform = PairsToTokens(prot_words=PROT_WORDS, apta_max_len=10, prot_max_len=8)

    assert not hasattr(transform, "prot_words_")

    transform.fit(X)

    assert transform.prot_words_ == filter_words(PROT_WORDS)
    assert transform.prot_words == PROT_WORDS


def test_transform_before_fit_raises():
    """transform without fit raises NotFittedError, not a bare AttributeError."""
    X = MoleculeLoader(data={"aptamer": [APTAMER], "protein": [PROTEIN]})
    transform = PairsToTokens(prot_words=PROT_WORDS, apta_max_len=10, prot_max_len=8)

    with pytest.raises(NotFittedError):
        transform.transform(X)


def test_fit_is_idempotent():
    """Refitting derives prot_words_ from the raw vocabulary, not the filtered one."""
    X = MoleculeLoader(data={"aptamer": [APTAMER], "protein": [PROTEIN]})
    transform = PairsToTokens(prot_words=PROT_WORDS, apta_max_len=10, prot_max_len=8)

    transform.fit(X)
    first = dict(transform.prot_words_)
    transform.fit(X)

    assert transform.prot_words_ == first
