"""Tests for the PairsToFeatures transform."""

__author__ = ["siddharth7113"]

import pandas as pd
import pytest

from pyaptamer.aptanet import PairsToFeatures
from pyaptamer.data import MoleculeLoader

APTAMER = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
PROTEIN = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


def test_pairs_to_features_from_moleculeloader():
    """A MoleculeLoader of aptamer/protein pairs becomes a numeric feature table."""
    X = MoleculeLoader(data={"aptamer": [APTAMER] * 3, "protein": [PROTEIN] * 3})
    Xt = PairsToFeatures(k=4).fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert len(Xt) == 3
    assert Xt.shape[1] > 1  # concatenated k-mer + PSeAAC features


def test_pairs_to_features_custom_columns():
    """Column names are configurable, not hardcoded to aptamer/protein."""
    X = MoleculeLoader(
        data={"aptamer_sequence": [APTAMER] * 2, "target_sequence": [PROTEIN] * 2}
    )
    transform = PairsToFeatures(
        aptamer_col="aptamer_sequence", protein_col="target_sequence"
    )
    Xt = transform.fit_transform(X)

    assert len(Xt) == 2


def test_pairs_to_features_rejects_non_moleculeloader():
    """Only a MoleculeLoader is accepted; a plain DataFrame is rejected."""
    X = pd.DataFrame({"aptamer": [APTAMER], "protein": [PROTEIN]})
    with pytest.raises(TypeError, match="only a MoleculeLoader"):
        PairsToFeatures().fit_transform(X)
