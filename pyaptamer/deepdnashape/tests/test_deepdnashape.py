__author__ = "prashantpandeygit"

import numpy as np
import pytest

from pyaptamer.deepdnashape import deepDNAshape

# test sequence
TEST_SEQ = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"


@pytest.fixture
def predictor():
    """Returns an instance of deepDNAshape for testing."""
    return deepDNAshape()


def test_invalid_feature(predictor):
    """Test an unknown DNA shape feature raises ValueError."""
    with pytest.raises(ValueError, match="Unknown feature"):
        predictor.predict("INVALID_FEATURE", TEST_SEQ)


@pytest.mark.parametrize("layer", [-1, 8, 10])
def test_invalid_layer(predictor, layer):
    """Test out of bound layer number raises ValueError."""
    with pytest.raises(ValueError, match="layer must be between 0 and 7"):
        predictor.predict("MGW", TEST_SEQ, layer=layer)


@pytest.mark.parametrize("feature", ["MGW", "ProT"])
def test_intrabase_model_shape(predictor, feature):
    """
    Test that intrabase features return an array of shape (N,)
    matching the length of the input sequence.
    """
    preds = predictor.predict(feature, TEST_SEQ)

    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.floating)
    assert preds.shape == (len(TEST_SEQ),)


@pytest.mark.parametrize("feature", ["Roll", "HelT"])
def test_interbase_model_shape(predictor, feature):
    """
    Test that interbase features return an array of shape (N-1,)
    reflecting steps between base pairs.
    """
    preds = predictor.predict(feature, TEST_SEQ)

    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.floating)
    assert preds.shape == (len(TEST_SEQ) - 1,)


@pytest.mark.parametrize("layer", [0, 4, 7])
def test_layer_prediction(predictor, layer):
    """Test if valid layer numbers produce predictions successfully."""
    feature = "MGW"
    preds = predictor.predict(feature, TEST_SEQ, layer=layer)
    assert preds.shape == (len(TEST_SEQ),)


def test_reverse_complement_invariance(predictor):
    """
    Verify that the predictor is invariant to DNA orientation due to its
    internal reverse-complement logic.
    """
    feature = "MGW"
    # sequence and its reverse complement
    seq = "ATGC"
    rev_comp_seq = "GCAT"

    res1 = predictor.predict(feature, seq)
    res2 = predictor.predict(feature, rev_comp_seq)

    # After flipping the results shourld match
    np.testing.assert_allclose(res1, res2[::-1], atol=1e-5)
