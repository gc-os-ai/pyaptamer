__author__ = ["prashantpandeygit", "Alleny244"]

import numpy as np
import pytest

from pyaptamer.deepdnashape import deepDNAshape

# test sequence
TEST_SEQ = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"

# Short sequence used for numerical reference checks.
REF_SEQ = "AGCTTAGCGT"

# Frozen Torch-port outputs for REF_SEQ at layer=4.
# These lock current numerical behavior in CI. Re-check against the
# original TensorFlow deepDNAshape if model weights or rescaling change.
_REF_PREDICTIONS = {
    "MGW": np.array(
        [
            5.26707745,
            4.5357666,
            4.36672401,
            4.82271385,
            5.50648785,
            5.84248638,
            5.25734043,
            5.0927515,
            5.19374323,
            5.48550797,
        ],
        dtype=np.float64,
    ),
    "ProT": np.array(
        [
            -9.56554794,
            -1.1265204,
            -1.65795779,
            -8.57664871,
            -10.11034775,
            -6.49536705,
            -1.33570004,
            -4.47787857,
            -11.63541126,
            -15.03390121,
        ],
        dtype=np.float64,
    ),
    "Roll": np.array(
        [
            -0.99402332,
            -3.35801697,
            -2.87414646,
            -3.15939999,
            5.54953861,
            -2.44367003,
            -2.12901092,
            4.13942862,
            -2.05449367,
        ],
        dtype=np.float64,
    ),
    "HelT": np.array(
        [
            31.53108978,
            37.94057465,
            32.00387573,
            34.9837265,
            34.65808868,
            31.51265907,
            37.09857941,
            33.07782745,
            34.76763916,
        ],
        dtype=np.float64,
    ),
}


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


@pytest.mark.parametrize("feature", ["MGW", "ProT", "Roll", "HelT"])
def test_reference_predictions(predictor, feature):
    """Predictions match frozen numerical reference values.

    Uses a short fixed sequence and layer=4. This catches accidental
    changes to encoding, graph building, rescaling, or weight loading.
    """
    expected = _REF_PREDICTIONS[feature]
    actual = predictor.predict(feature, REF_SEQ, layer=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
