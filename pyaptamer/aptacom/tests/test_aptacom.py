__author__ = ["siddharth7113"]

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptacom import AptaComClassifier, AptaComPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_feature_extraction(monkeypatch):
    """Mock the heavy PyBioMed and rust_sasa_python dependencies.

    Since these are optional/heavy, we mock the underlying extraction
    functions to return simple deterministic arrays for pipeline testing.
    """
    import pyaptamer.utils._aptacom_utils as aptacom_utils

    def mock_dna(dna):
        return np.ones(1354, dtype=np.float32) * len(dna)

    def mock_protein(prot):
        return np.ones(9920, dtype=np.float32) * len(prot)

    def mock_sasa(pdb_path):
        return np.ones(20, dtype=np.float32)

    monkeypatch.setattr(aptacom_utils, "_dna_pybiomed", mock_dna)
    monkeypatch.setattr(aptacom_utils, "_protein_pybiomed", mock_protein)
    monkeypatch.setattr(aptacom_utils, "_residue_exposure", mock_sasa)


# ---------------------------------------------------------------------------
# Sample Sequences
# ---------------------------------------------------------------------------
APT_SEQ = "AGTCGATGGCTGAGGGATCGATG"
TRG_SEQ = (
    "MWLGRRALCALVLLLACASLGLLYASTRDAPGLRLPLAPWAPPQSPRRVTLTGEG"
    "QADLTLVSLDESQMAKHRLLFFKHRLQCMTSQ"
)

SS_DOT = "...(((....)))...((....))"  # synthetic dot-bracket (len 24)


# ---------------------------------------------------------------------------
# Pipeline-level tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("apt_seq, trg_seq", [(APT_SEQ, TRG_SEQ)], ids=["default_seqs"])
def test_pipeline_fit_and_predict_classification(apt_seq, trg_seq):
    """Pipeline produces valid binary labels with correct shape."""
    pipe = AptaComPipeline()

    X = [(apt_seq, trg_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("apt_seq, trg_seq", [(APT_SEQ, TRG_SEQ)], ids=["default_seqs"])
def test_pipeline_fit_and_predict_proba(apt_seq, trg_seq):
    """Pipeline probability estimates have correct shape and range."""
    pipe = AptaComPipeline()

    X = [(apt_seq, trg_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)

    assert proba.shape == (40, 2)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize(
    "apt_seq, trg_seq, ss",
    [(APT_SEQ, TRG_SEQ, SS_DOT)],
    ids=["with_ss"],
)
def test_pipeline_with_ss_features(apt_seq, trg_seq, ss):
    """Pipeline works when secondary-structure features are provided."""
    pipe = AptaComPipeline(features=["aptamer", "target", "ss"])

    X = [(apt_seq, trg_seq, ss) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Estimator sklearn compatibility
# ---------------------------------------------------------------------------


@parametrize_with_checks(
    estimators=[AptaComClassifier(n_estimators=2, fs_n_estimators=2)]
)
def test_sklearn_compatible_estimator(estimator, check):
    """Run scikit-learn's compatibility checks on the AptaComClassifier."""
    check(estimator)


# ---------------------------------------------------------------------------
# Estimator unit tests
# ---------------------------------------------------------------------------


def test_classifier_fit_predict():
    """AptaComClassifier basic fit/predict on random numeric data."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 20)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    clf = AptaComClassifier(random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


def test_classifier_predict_proba():
    """AptaComClassifier predict_proba returns valid probabilities."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 20)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    clf = AptaComClassifier(random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (40, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_no_feature_selection():
    """AptaComClassifier works with feature selection disabled."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    clf = AptaComClassifier(use_feature_selection=False, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert preds.shape == (40,)
