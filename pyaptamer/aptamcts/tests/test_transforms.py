"""Tests for the AptaMCTS PairsToFeatures transform."""

__author__ = ["aditi-dsi"]

import pandas as pd
import pytest

from pyaptamer.aptamcts._transforms import PairsToFeatures
from pyaptamer.data import MoleculeLoader

APTAMER = "AGCUUAGCGUACAGCUUAAAAGGGUUUCCCCUGCCCGCGTAC"
PROTEIN = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


def test_pairs_to_features_from_moleculeloader():
    """A MoleculeLoader of aptamer/protein pairs becomes a numeric feature table."""
    X = MoleculeLoader(data={"aptamer": [APTAMER] * 3, "protein": [PROTEIN] * 3})
    Xt = PairsToFeatures(rna_k=1, prot_k=1).fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert len(Xt) == 3
    assert Xt.shape[1] == 11


def test_pairs_to_features_custom_columns():
    """Column names are configurable, not hardcoded to aptamer/protein."""
    X = MoleculeLoader(
        data={"aptamer_sequence": [APTAMER] * 2, "target_sequence": [PROTEIN] * 2}
    )
    transform = PairsToFeatures(
        rna_k=1,
        prot_k=1,
        aptamer_col="aptamer_sequence",
        protein_col="target_sequence",
    )
    Xt = transform.fit_transform(X)

    assert len(Xt) == 2
    assert Xt.shape[1] == 11


def test_pairs_to_features_rejects_non_moleculeloader():
    """Only a MoleculeLoader is accepted, a plain DataFrame is rejected."""
    X = pd.DataFrame({"aptamer": [APTAMER], "protein": [PROTEIN]})

    with pytest.raises(TypeError, match="only a MoleculeLoader"):
        PairsToFeatures().fit_transform(X)
