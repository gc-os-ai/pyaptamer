__author__ = "satvshr"

import numpy as np
import pytest
import torch
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

from pyaptamer.deepaptamer import (
    DeepAptamerClassifier,
    DeepAptamerNN,
    DeepAptamerPipeline,
)


@pytest.mark.parametrize(
    "seqs",
    [
        "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTCCG",
        [
            "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC",
            "TGCATGCTAGCTAGCTAGCTAGCTAGCTAGCGCTA",
        ],
    ],
)
def test_pipeline_predict_shapes(seqs):
    """
    Test if DeepAptamerPipeline outputs valid ranked predictions.

    Raises
    ------
    AssertionError
        If prediction scores are not sorted in descending order.
    """
    model = DeepAptamerNN()
    pipe = DeepAptamerPipeline(model=model, device="cpu")

    ranked = pipe.predict(seqs)

    # Ensure sorted in descending order
    scores = [item["score"] for item in ranked]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


SEQ_LEN = 35
SHAPE_LEN = 126


def _make_features(n, seed=0):
    """Random one-hot sequences followed by random shape vectors."""
    rng = np.random.default_rng(seed)
    ohe = np.eye(4)[rng.integers(0, 4, size=(n, SEQ_LEN))].reshape(n, -1)
    shape = rng.standard_normal((n, SHAPE_LEN))
    return np.hstack([ohe, shape]).astype(np.float32)


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
    """X without room for the shape vector raises ValueError."""
    X = np.zeros((10, 4 * SEQ_LEN), dtype=np.float32)
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="one-hot features"):
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
