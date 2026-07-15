__author__ = ["prashantpandeygit", "Alleny244"]

import numpy as np
import pandas as pd
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


def _frame(*seqs):
    """Build a univariate DataFrame of DNA sequences."""
    return pd.DataFrame({"seq": list(seqs)})


def _values(Xt):
    """Return the first row of a transform output without trailing NaNs."""
    row = Xt.iloc[0].to_numpy(dtype=np.float64)
    if np.isnan(row).any():
        return row[~np.isnan(row)]
    return row


def test_invalid_feature():
    """Test an unknown DNA shape feature raises ValueError."""
    with pytest.raises(ValueError, match="Unknown feature"):
        deepDNAshape(feature="INVALID_FEATURE")


@pytest.mark.parametrize("layer", [-1, 8, 10])
def test_invalid_layer(layer):
    """Test out of bound layer number raises ValueError."""
    with pytest.raises(ValueError, match="layer must be between 0 and 7"):
        deepDNAshape(feature="MGW", layer=layer)


def test_fit_returns_self():
    """Empty fit follows the estimator contract and returns self."""
    est = deepDNAshape(feature="MGW")
    assert est.fit(_frame(TEST_SEQ)) is est


@pytest.mark.parametrize("feature", ["MGW", "ProT"])
def test_intrabase_model_shape(feature):
    """
    Test that intrabase features return N values matching sequence length.
    """
    Xt = deepDNAshape(feature=feature).fit_transform(_frame(TEST_SEQ))
    preds = _values(Xt)

    assert isinstance(Xt, pd.DataFrame)
    assert np.issubdtype(preds.dtype, np.floating)
    assert preds.shape == (len(TEST_SEQ),)


@pytest.mark.parametrize("feature", ["Roll", "HelT"])
def test_interbase_model_shape(feature):
    """
    Test that interbase features return N-1 values between base pairs.
    """
    Xt = deepDNAshape(feature=feature).fit_transform(_frame(TEST_SEQ))
    preds = _values(Xt)

    assert isinstance(Xt, pd.DataFrame)
    assert np.issubdtype(preds.dtype, np.floating)
    assert preds.shape == (len(TEST_SEQ) - 1,)


@pytest.mark.parametrize("layer", [0, 4, 7])
def test_layer_prediction(layer):
    """Test if valid layer numbers produce predictions successfully."""
    Xt = deepDNAshape(feature="MGW", layer=layer).fit_transform(_frame(TEST_SEQ))
    assert _values(Xt).shape == (len(TEST_SEQ),)


def test_reverse_complement_invariance():
    """
    Verify that the transformer is invariant to DNA orientation due to its
    internal reverse-complement logic.
    """
    est = deepDNAshape(feature="MGW")
    res1 = _values(est.fit_transform(_frame("ATGC")))
    res2 = _values(est.fit_transform(_frame("GCAT")))
    np.testing.assert_allclose(res1, res2[::-1], atol=1e-5)


def test_batch_nan_padding():
    """Shorter sequences are right-padded with NaN in a batch."""
    short = "ATGC"
    long = TEST_SEQ
    Xt = deepDNAshape(feature="MGW").fit_transform(_frame(short, long))

    assert Xt.shape == (2, len(long))
    assert np.isnan(Xt.iloc[0, len(short) :]).all()
    assert not np.isnan(Xt.iloc[0, : len(short)]).any()
    assert not np.isnan(Xt.iloc[1].to_numpy()).any()


@pytest.mark.parametrize("feature", ["MGW", "ProT", "Roll", "HelT"])
def test_reference_predictions(feature):
    """Predictions match frozen numerical reference values.

    Uses a short fixed sequence and layer=4. This catches accidental
    changes to encoding, graph building, rescaling, or weight loading.
    """
    expected = _REF_PREDICTIONS[feature]
    actual = _values(
        deepDNAshape(feature=feature, layer=4).fit_transform(_frame(REF_SEQ))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
