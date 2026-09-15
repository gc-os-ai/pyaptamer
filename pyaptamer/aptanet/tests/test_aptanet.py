__author__ = ["nennomp", "satvshr", "siddharth7113"]


import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptanet import AptaNetClassifier, AptaNetPipeline, AptaNetRegressor
from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.encode import KMerFrequencies, PSeAAC

params = [
    (
        "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    )
]


def _make_loader(aptamer_seq, protein_seq, n):
    """Build a MoleculeLoader of n identical aptamer/protein pairs."""
    return MoleculeLoader(
        data={"aptamer": [aptamer_seq] * n, "protein": [protein_seq] * n}
    )


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_and_predict_classification(aptamer_seq, protein_seq):
    """
    Test if Pipeline predictions are valid class labels and shape matches input
    for classification.
    """
    pipe = AptaNetPipeline(k=4)

    X_raw = _make_loader(aptamer_seq, protein_seq, 40)
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X_raw, y)
    preds = pipe.predict(X_raw)

    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_and_predict_proba(aptamer_seq, protein_seq):
    """
    Test if Pipeline probability estimates predictions returns floats and shape matches
    input.
    """
    pipe = AptaNetPipeline()

    X_raw = _make_loader(aptamer_seq, protein_seq, 40)
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X_raw, y)
    preds = pipe.predict_proba(X_raw)

    assert preds.shape == (40, 2)
    assert preds.dtype == np.float32
    assert np.all((preds >= 0) & (preds <= 1))


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_fit_and_predict_regression(aptamer_seq, protein_seq):
    """
    Test if Pipeline predictions are valid floats and shape matches input
    for regression.
    """
    pipe = AptaNetPipeline(estimator=AptaNetRegressor())

    X_raw = _make_loader(aptamer_seq, protein_seq, 40)
    y = np.linspace(0, 1, 40).astype(np.float32)

    pipe.fit(X_raw, y)
    preds = pipe.predict(X_raw)

    assert preds.shape == (40,)
    assert np.issubdtype(preds.dtype, np.floating)


def _aptanet_features():
    """The feature step AptaNetPipeline builds, for use on its own."""
    return ColumnTransformer(
        [
            ("aptamer", KMerFrequencies(), ["aptamer"]),
            ("protein", PSeAAC(), ["protein"]),
        ]
    )


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_fasta_sourced_loader_matches_in_memory(tmp_path, aptamer_seq, protein_seq):
    """A FASTA-sourced loader yields the same features as the in-memory loader."""
    fasta = tmp_path / "library.fasta"
    fasta.write_text(f">apt1\n{aptamer_seq}\n>apt2\n{aptamer_seq}\n")

    from_file = MoleculeLoader(
        data={"aptamer": [str(fasta)], "protein": [protein_seq]}, tiling="samples"
    )
    in_memory = _make_loader(aptamer_seq, protein_seq, 2)

    features = _aptanet_features()
    feats_file = features.fit_transform(from_file.to_dataframe())
    feats_mem = features.fit_transform(in_memory.to_dataframe())

    assert feats_file.shape == feats_mem.shape == (2, 690)
    assert np.allclose(feats_file, feats_mem)


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_custom_column_names(aptamer_seq, protein_seq):
    """aptamer_col and protein_col select the columns; nothing is hardcoded."""
    X = MoleculeLoader(data={"apt": [aptamer_seq] * 40, "target": [protein_seq] * 40})
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    pipe = AptaNetPipeline(aptamer_col="apt", protein_col="target").fit(X, y)
    assert pipe.predict(X).shape == (40,)


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_accepts_dataframe(aptamer_seq, protein_seq):
    """A DataFrame with the aptamer and protein columns is accepted like a loader."""
    X = _make_loader(aptamer_seq, protein_seq, 40).to_dataframe()
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    pipe = AptaNetPipeline().fit(X, y)
    assert pipe.predict(X).shape == (40,)
    assert pipe.pipeline_["features"].transform(X).shape == (40, 690)


@parametrize_with_checks(
    estimators=[AptaNetClassifier(), AptaNetRegressor()],
    expected_failed_checks={
        "check_pipeline_consistency": "estimator is non-deterministic"
    },
)
def test_sklearn_compatible_estimator(estimator, check):
    """
    Run scikit-learn's compatibility checks on the AptaNetClassifier.
    """
    check(estimator)
