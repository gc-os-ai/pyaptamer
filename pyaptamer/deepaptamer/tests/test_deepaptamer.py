__author__ = ["satvshr", "geetu040"]

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.dummy import DummyClassifier
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

from pyaptamer.data import MoleculeLoader
from pyaptamer.deepaptamer import (
    DeepAptamerClassifier,
    DeepAptamerFeatures,
    DeepAptamerPipeline,
)

APTAMERS = [
    "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC",
    "TGCATGCTAGCTAGCTAGCTAGCTAGCTAGCGCTA",
]


def _make_features(n, seed=0):
    """Random one-hot 35-mers followed by random 126-value shape vectors."""
    rng = np.random.default_rng(seed)
    ohe = np.eye(4)[rng.integers(0, 4, size=(n, 35))].reshape(n, -1)
    shape = rng.standard_normal((n, 126))
    return np.hstack([ohe, shape]).astype(np.float32)


@pytest.mark.parametrize("full_dna_shape, width", [(False, 266), (True, 278)])
def test_features_width(full_dna_shape, width):
    """The width is 4 * 35 one-hot values plus 126 shape values, or 138 if full."""
    X = pd.DataFrame({"aptamer": APTAMERS})
    Xt = DeepAptamerFeatures(full_dna_shape=full_dna_shape).fit_transform(X)

    assert Xt.shape == (2, width)
    assert Xt.index.equals(X.index)
    # "A" -> [1, 0, 0, 0], "G" -> [0, 0, 0, 1]
    assert Xt.iloc[0, :8].tolist() == [1, 0, 0, 0, 0, 0, 0, 1]


def test_features_pad_shorter_sequences():
    """Sequences shorter than seq_len are padded, so mixed lengths share a width."""
    X = pd.DataFrame({"aptamer": [APTAMERS[0], APTAMERS[1][:20]]})
    Xt = DeepAptamerFeatures().fit_transform(X)

    assert Xt.shape == (2, 266)
    assert not Xt.isna().any().any()
    # padding "N" is encoded as all zeros
    assert (Xt.iloc[1, 4 * 20 : 4 * 35] == 0).all()


def test_features_reject_too_long_sequences():
    """Sequences longer than seq_len raise ValueError."""
    X = pd.DataFrame({"aptamer": APTAMERS})
    with pytest.raises(ValueError, match="exceeds"):
        DeepAptamerFeatures(seq_len=20).fit_transform(X)


def test_pipeline_fit_and_predict():
    """Pipeline predictions are valid labels and probabilities with matching shape."""
    X = MoleculeLoader(data={"aptamer": APTAMERS * 2})
    y = np.array([0, 1] * 2)
    pipe = DeepAptamerPipeline(
        estimator=DeepAptamerClassifier(max_epochs=2, random_state=0)
    ).fit(X, y)

    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)

    assert preds.shape == (4,)
    assert set(preds).issubset({0, 1})
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)


def test_pipeline_full_dna_shape_reaches_features():
    """full_dna_shape is passed on to the features step, giving 4 * 35 + 138 values."""
    X = MoleculeLoader(data={"aptamer": APTAMERS * 2})
    y = np.array([0, 1] * 2)
    pipe = DeepAptamerPipeline(full_dna_shape=True, estimator=DummyClassifier())
    pipe.fit(X, y)

    features = pipe.pipeline_["features"]
    assert features.transform(X.to_dataframe()).shape == (4, 278)


def test_pipeline_sets_seq_len_on_estimator():
    """The pipeline's seq_len overrides the seq_len of a custom estimator."""
    X = MoleculeLoader(data={"aptamer": APTAMERS * 2})
    y = np.array([0, 1] * 2)
    pipe = DeepAptamerPipeline(
        seq_len=40, estimator=DeepAptamerClassifier(max_epochs=1, random_state=0)
    ).fit(X, y)

    assert pipe.pipeline_["clf"].seq_len == 40


def test_pipeline_accepts_dataframe_and_custom_column():
    """A DataFrame is accepted like a loader and aptamer_col selects the column."""
    X = pd.DataFrame({"apt": APTAMERS * 2})
    y = np.array([0, 1] * 2)
    pipe = DeepAptamerPipeline(aptamer_col="apt", estimator=DummyClassifier())

    assert pipe.fit(X, y).predict(X).shape == (4,)


def test_pipeline_rejects_other_input():
    """Input that is neither a MoleculeLoader nor a DataFrame raises TypeError."""
    with pytest.raises(
        TypeError, match="MoleculeLoader instance or a pandas DataFrame"
    ):
        DeepAptamerPipeline().fit(APTAMERS, np.array([0, 1]))


def test_classifier_fit_predict():
    """predict returns labels from classes_ and predict_proba rows sum to 1."""
    X = _make_features(40)
    y = np.array(["no"] * 20 + ["yes"] * 20)
    clf = DeepAptamerClassifier(max_epochs=2, random_state=0).fit(X, y)

    proba = clf.predict_proba(X)
    preds = clf.predict(X)

    assert list(clf.classes_) == ["no", "yes"]
    assert proba.shape == (40, 2)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert preds.shape == (40,)
    assert set(preds).issubset({"no", "yes"})


def test_classifier_rejects_multiclass():
    """Targets with more than two classes raise ValueError."""
    X = _make_features(30)
    y = np.repeat([0, 1, 2], 10)
    with pytest.raises(ValueError, match="Only binary classification"):
        DeepAptamerClassifier(max_epochs=1).fit(X, y)


def test_classifier_rejects_too_few_features():
    """A shape vector one value shorter than the 119 the network needs raises."""
    X = np.zeros((10, 4 * 35 + 118), dtype=np.float32)
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="at least 119 shape features"):
        DeepAptamerClassifier(max_epochs=1).fit(X, y)


def test_classifier_fit_leaves_global_torch_rng_untouched():
    """fit with random_state does not change the global Torch RNG state."""
    X = _make_features(20)
    y = np.array([0, 1] * 10)
    before = torch.get_rng_state()
    DeepAptamerClassifier(max_epochs=1, random_state=0).fit(X, y)

    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "check",
    [
        check_parameters_default_constructible,
        check_get_params_invariance,
        check_set_params,
        check_no_attributes_set_in_init,
    ],
)
def test_classifier_sklearn_param_checks(check):
    """The scikit-learn parameter checks that do not depend on the input data."""
    check("DeepAptamerClassifier", DeepAptamerClassifier())
