__author__ = ["prashantpandeygit", "Alleny244"]

import re

import numpy as np
import pandas as pd
import pytest
import torch

from pyaptamer.data import MoleculeLoader
from pyaptamer.deepdnashape import DeepDNAShape, _predictor
from pyaptamer.deepdnashape._model import AvgFeatures

# test sequence
TEST_SEQ = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"

# Short sequence used for numerical reference checks.
REF_SEQ = "AGCTTAGCGT"

# Frozen Torch-port outputs for REF_SEQ at layer=4. They match the original
# TensorFlow deepDNAshape (tensorflow 2.12) within 4e-6. Re-check against it
# if model weights or rescaling change.
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
        DeepDNAShape(feature="INVALID_FEATURE")


@pytest.mark.parametrize("layer", [-1, 8, 10])
def test_invalid_layer(layer):
    """Test out of bound layer number raises ValueError."""
    with pytest.raises(ValueError, match="layer must be between 0 and 7"):
        DeepDNAShape(feature="MGW", layer=layer)


def test_allowed_bases_pass_check():
    """A, C, G, T, and N are accepted."""
    Xt = DeepDNAShape(feature="MGW").fit_transform(_frame("ACGTN"))
    assert Xt.shape == (1, 5)


@pytest.mark.parametrize(
    ("seq", "bad"),
    [
        ("ATGX", "['X']"),
        ("atgc", "['a', 'c', 'g', 't']"),
    ],
)
def test_invalid_base(seq, bad):
    """Bases outside A, C, G, T, N raise ValueError naming them."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Invalid bases {bad} in '{seq}'. Allowed bases are A, C, G, T, N."
        ),
    ):
        DeepDNAShape(feature="MGW").fit_transform(_frame(seq))


def test_bag_tiling_raises(tmp_path):
    """A loader row holding several reads raises TypeError pointing to samples."""
    fasta = tmp_path / "reads.fasta"
    fasta.write_text(">a\nACGTACGTAC\n>b\nTTTTGGGGCC\n")
    X = MoleculeLoader(data={"seq": [str(fasta)]}, tiling="bag")
    with pytest.raises(TypeError, match='tiling="samples"'):
        DeepDNAShape(feature="MGW").fit_transform(X)


def test_fit_returns_self():
    """Empty fit follows the estimator contract and returns self."""
    est = DeepDNAShape(feature="MGW")
    assert est.fit(_frame(TEST_SEQ)) is est


@pytest.mark.parametrize(
    ("feature", "layer", "expected_len"),
    [
        ("MGW", 4, len(TEST_SEQ)),  # intrabase
        ("ProT", 4, len(TEST_SEQ)),  # intrabase
        ("Roll", 4, len(TEST_SEQ) - 1),  # interbase
        ("HelT", 4, len(TEST_SEQ) - 1),  # interbase
        ("MGW", 0, len(TEST_SEQ)),  # layer smoke
        ("MGW", 7, len(TEST_SEQ)),  # layer smoke
    ],
)
def test_output_shape(feature, layer, expected_len):
    """Output length matches feature kind; valid layers produce floats."""
    Xt = DeepDNAShape(feature=feature, layer=layer).fit_transform(_frame(TEST_SEQ))
    preds = _values(Xt)

    assert isinstance(Xt, pd.DataFrame)
    assert np.issubdtype(preds.dtype, np.floating)
    assert preds.shape == (expected_len,)


@pytest.mark.parametrize(
    ("feature", "sign"),
    [
        ("MGW", 1),
        ("Shear", -1),
    ],
)
def test_reverse_complement(feature, sign):
    """A sequence matches its reverse complement, negated when flip_rev is set.

    ``GCAT`` is the reverse complement of ``ATGC``. Symmetric features such
    as ``MGW`` match directly. Antisymmetric features such as ``Shear``
    match only after negation.
    """
    est = DeepDNAShape(feature=feature)
    forward = _values(est.fit_transform(_frame("ATGC")))
    reverse = _values(est.fit_transform(_frame("GCAT")))
    np.testing.assert_allclose(forward, sign * reverse[::-1], atol=1e-5)
    if sign < 0:
        assert not np.allclose(forward, reverse[::-1], atol=1e-5)


def test_batch_nan_padding():
    """Shorter sequences are right-padded with NaN in a batch."""
    short = "ATGC"
    long = TEST_SEQ
    Xt = DeepDNAShape(feature="MGW").fit_transform(_frame(short, long))

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
        DeepDNAShape(feature=feature, layer=4).fit_transform(_frame(REF_SEQ))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_avg_features_pads_to_next_multiple():
    """AvgFeatures pads up to the next multiple of target_features."""
    layer = AvgFeatures(target_features=5, filter_size=64)
    assert layer.pad_amount == 1
    assert layer.group_size == 13

    x = torch.randn(3, 64)
    out = layer(x)
    assert out.shape == (3 * 5,)


def _random_seqs(n, length, seed=0):
    """Return ``n`` random DNA strings of the given length."""
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), size=length)) for _ in range(n)]


def _one_at_a_time(est, seqs):
    """Transform each sequence in its own call and stack the rows."""
    return np.vstack([est.fit_transform(_frame(seq)).to_numpy() for seq in seqs])


@pytest.mark.parametrize("layer", [2, 4])
@pytest.mark.parametrize("feature", ["MGW", "ProT", "Roll", "HelT"])
def test_batch_matches_one_at_a_time(feature, layer):
    """Scoring sequences together gives the same values as one at a time."""
    seqs = _random_seqs(6, 20)
    est = DeepDNAShape(feature=feature, layer=layer)
    batched = est.fit_transform(_frame(*seqs)).to_numpy()
    np.testing.assert_allclose(batched, _one_at_a_time(est, seqs), rtol=1e-5, atol=1e-5)


def test_batch_single_sequence():
    """A batch of one sequence works and keeps its shape."""
    Xt = DeepDNAShape(feature="MGW").fit_transform(_frame(REF_SEQ))
    assert Xt.shape == (1, len(REF_SEQ))
    assert not Xt.isna().any().any()


def test_batch_mixed_lengths_keep_order_and_padding():
    """Rows keep the input order and NaN padding when lengths are mixed."""
    seqs = ["ATGCATGCAT", TEST_SEQ, "GGCCA", "TTAGCGTACA", "A"]
    est = DeepDNAShape(feature="MGW")
    Xt = est.fit_transform(_frame(*seqs))

    assert Xt.shape == (len(seqs), len(TEST_SEQ))
    for i, seq in enumerate(seqs):
        alone = est.fit_transform(_frame(seq)).to_numpy()[0]
        np.testing.assert_allclose(Xt.iloc[i, : len(seq)], alone, rtol=1e-5, atol=1e-5)
        assert np.isnan(Xt.iloc[i, len(seq) :]).all()


def test_batch_keeps_index():
    """The output keeps the index of the input."""
    X = pd.DataFrame({"seq": ["ATGC", "GGCCAT", "TTAA"]}, index=[7, 3, 5])
    Xt = DeepDNAShape(feature="MGW").fit_transform(X)
    assert list(Xt.index) == [7, 3, 5]


def test_batch_larger_than_batch_size():
    """More sequences than batch_size give the same values as a single pass."""
    seqs = _random_seqs(7, 15)
    X = _frame(*seqs)
    whole = DeepDNAShape(feature="Roll").fit_transform(X)
    chunked = DeepDNAShape(feature="Roll", batch_size=3).fit_transform(X)
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(batch_size):
    """A batch_size below 1 raises ValueError."""
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        DeepDNAShape(feature="MGW", batch_size=batch_size)


def test_model_loaded_once_across_transforms(monkeypatch):
    """Repeated transforms on one instance download and load weights once."""
    calls = []
    original = _predictor.hf_hub_download

    def counting(*args, **kwargs):
        calls.append(kwargs["filename"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_predictor, "hf_hub_download", counting)
    est = DeepDNAShape(feature="MGW")
    for _ in range(3):
        est.fit_transform(_frame(REF_SEQ))
    assert calls == ["MGW.pt"]
