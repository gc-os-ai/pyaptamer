__author__ = ["nennomp", "satvshr", "siddharth7113"]


import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptanet import AptaNetClassifier, AptaNetPipeline, AptaNetRegressor
from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.encode.tests._pseaac_solution import solution

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


def _fitted_features(X):
    """The fitted feature step of an AptaNetPipeline, without training the network."""
    y = np.zeros(len(X.to_dataframe()), dtype=np.float32)
    y[: len(y) // 2] = 1
    pipe = AptaNetPipeline(estimator=DummyClassifier()).fit(X, y)
    return pipe.pipeline_["features"]


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_fasta_sourced_loader_matches_in_memory(tmp_path, aptamer_seq, protein_seq):
    """A FASTA-sourced loader yields the same features as the in-memory loader."""
    fasta = tmp_path / "library.fasta"
    fasta.write_text(f">apt1\n{aptamer_seq}\n>apt2\n{aptamer_seq}\n")

    from_file = MoleculeLoader(
        data={"aptamer": [str(fasta)], "protein": [protein_seq]}, tiling="samples"
    )
    in_memory = _make_loader(aptamer_seq, protein_seq, 2)

    features = _fitted_features(in_memory)
    feats_file = features.transform(from_file.to_dataframe())
    feats_mem = features.transform(in_memory.to_dataframe())

    assert feats_file.shape == feats_mem.shape == (2, 690)
    assert np.allclose(feats_file, feats_mem)


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_custom_column_names(aptamer_seq, protein_seq):
    """aptamer_col and protein_col select the columns; nothing is hardcoded."""
    X = MoleculeLoader(data={"apt": [aptamer_seq] * 4, "target": [protein_seq] * 4})
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    pipe = AptaNetPipeline(
        aptamer_col="apt", protein_col="target", estimator=DummyClassifier()
    ).fit(X, y)
    assert pipe.predict(X).shape == (4,)


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_pseaac_is_pinned_to_aptanet_reference(aptamer_seq, protein_seq):
    """The protein step reproduces the AptaNet reference PSeAAC vector.

    The pipeline is fitted with a DummyClassifier so that the test reaches the
    fitted ColumnTransformer without training the network.
    """
    features = _fitted_features(_make_loader(aptamer_seq, protein_seq, 4))
    protein_step = features.named_transformers_["protein"]
    ref = pd.DataFrame({"protein": ["ACDFFKKIIKKLLMMNNPPQQQRRRRIIIIRRR"]})
    np.testing.assert_allclose(
        protein_step.transform(ref).to_numpy()[0], solution, atol=1e-3
    )


@pytest.mark.parametrize("aptamer_seq, protein_seq", params)
def test_pipeline_accepts_dataframe(aptamer_seq, protein_seq):
    """A DataFrame with the aptamer and protein columns is accepted like a loader."""
    X = _make_loader(aptamer_seq, protein_seq, 4).to_dataframe()
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    pipe = AptaNetPipeline(estimator=DummyClassifier()).fit(X, y)
    assert pipe.predict(X).shape == (4,)


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
