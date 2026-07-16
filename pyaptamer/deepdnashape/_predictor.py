"""DeepDNAshape transform for DNA shape feature prediction."""

__author__ = ["prashantpandeygit", "Alleny244"]
__all__ = ["deepDNAshape"]

import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from pyaptamer.trafos.base import BaseTransform

from ._model import DNAModel

_HF_REPO_ID = "parkneurals/deepdnashape"
_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "_params.json")

_REV_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

_INTRABASE_FEATURES = frozenset(
    {
        "Shear",
        "Stretch",
        "Stagger",
        "Buckle",
        "ProT",
        "Opening",
        "MGW",
        "EP",
        "Shear-FL",
        "Stretch-FL",
        "Stagger-FL",
        "Buckle-FL",
        "ProT-FL",
        "Opening-FL",
        "MGW-FL",
    }
)
_INTERBASE_FEATURES = frozenset(
    {
        "Shift",
        "Slide",
        "Rise",
        "Tilt",
        "Roll",
        "HelT",
        "Shift-FL",
        "Slide-FL",
        "Rise-FL",
        "Tilt-FL",
        "Roll-FL",
        "HelT-FL",
    }
)
_ALL_FEATURES = _INTRABASE_FEATURES | _INTERBASE_FEATURES

_X_AXIS_FEATURES = frozenset({"Shift", "Tilt", "Shear", "Buckle"})


def _get_bases_mapping():
    """Build one-hot encoding lookup tables for mono- and di-nucleotides.

    Single bases (A, T, C, G) are mapped to length-4 one-hot vectors.
    Di-nucleotide pairs are mapped to length-16 one-hot vectors (the
    outer product of single-base encodings). Unknown bases (N)
    are encoded as uniform distributions.

    Returns
    -------
    tuple of (dict, dict)
        (mono, di) where mono maps single-character bases to
        np.ndarray of shape (4,) and di maps 2-tuples of
        bases to np.ndarray of shape (16,).
    """
    bases = ["T", "G", "C", "A"]
    n_bases = len(bases)

    mono = {}
    for i, base in enumerate(bases):
        vec = np.zeros(n_bases)
        vec[i] = 1
        mono[base] = vec
    mono["N"] = np.ones(n_bases, dtype=float) / n_bases

    di = {("N", "N"): np.ones(n_bases * n_bases, dtype=float) / n_bases / n_bases}
    for i, pair in enumerate(itertools.product(bases, repeat=2)):
        vec = np.zeros(n_bases * n_bases)
        vec[i] = 1
        di[pair] = vec

    for bp in bases:
        di[(bp, "N")] = np.zeros(n_bases * n_bases)
        di[("N", bp)] = np.zeros(n_bases * n_bases)
        for bp2 in bases:
            di[(bp, "N")] += di[(bp, bp2)]
            di[("N", bp)] += di[(bp2, bp)]
        di[(bp, "N")] /= n_bases
        di[("N", bp)] /= n_bases

    return mono, di


def _rescale(predictions, params):
    """Rescale raw model predictions to original value range.

    Parameters
    ----------
    predictions : np.ndarray
        Raw normalized predictions from the model.
    params : dict
        Scaling parameters for this feature (keys depend on the
        normalization method used during training).

    Returns
    -------
    np.ndarray
        Predictions rescaled to the original value range.
    """
    method = params["method"]
    if method == "minmax":
        return predictions * (params["max"] - params["min"]) + params["min"]
    if method == "minmax2":
        return (predictions + 1) * (params["max"] - params["min"]) / 2 + params["min"]
    if method == "sin":
        return np.arcsin(predictions) / np.pi * 180.0
    if method == "standard":
        return predictions * params["std"] + params["mean"]
    return predictions * params["percentile_range"] + params["median"]


def _as_sequence_str(value):
    """Coerce a DataFrame cell to a DNA sequence string."""
    if isinstance(value, str):
        return value
    return "".join(value)


class deepDNAshape(BaseTransform):  # noqa: N801
    """Transform DNA sequences into structural shape feature values.

    Given DNA strings (A/T/C/G, optionally N), this transformer predicts
    numeric shape properties such as Minor Groove Width (``MGW``),
    propeller twist (``ProT``), helical twist (``HelT``), and roll
    (``Roll``).

    There are two kinds of features:

    - **Intrabase** features (e.g. ``MGW``, ``ProT``): one value per
      base, so a sequence of length ``N`` yields ``N`` values.
    - **Interbase** features (e.g. ``Roll``, ``HelT``): one value per
      step between bases, so a sequence of length ``N`` yields
      ``N - 1`` values.

    Variable-length outputs are right-padded with ``NaN`` so every row
    in a batch has the same width.

    Fitting is empty: model weights are pretrained and downloaded from
    Hugging Face on first ``transform``.

    Internally each sequence is one-hot encoded, padded with two ``N``
    bases on each side, and scored in both forward and reverse-complement
    orientations. The two predictions are averaged.

    Original author: Jinsen Li.
    Original implementation: https://github.com/JinsenLi/deepDNAshape
    License: BSD-3-Clause

    Parameters
    ----------
    feature : str, default="MGW"
        Structural property to predict. Must be a supported feature name.
    layer : int, default=4
        Message-passing depth to read (``0`` = initial convolution only,
        ``7`` = deepest layer).

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.deepdnashape import deepDNAshape
    >>> X = pd.DataFrame({"seq": ["AAGGTAGT"]})
    >>> mgw = deepDNAshape(feature="MGW").fit_transform(X)
    >>> roll = deepDNAshape(feature="Roll").fit_transform(X)
    """

    _tags = {
        "authors": ["prashantpandeygit", "Alleny244"],
        "maintainers": ["Alleny244"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
        "capability:y": False,
    }

    def __init__(self, feature="MGW", layer=4):
        self.feature = feature
        self.layer = layer
        super().__init__()

        if self.feature not in _ALL_FEATURES:
            raise ValueError(
                f"Unknown feature. Must be one of {sorted(_ALL_FEATURES)}."
            )
        if not 0 <= self.layer <= 7:
            raise ValueError(f"layer must be between 0 and 7, got {self.layer}.")

        self._mono, self._di = _get_bases_mapping()
        with open(_PARAMS_PATH) as f:
            self._params = json.load(f)
        self._model = None

    def _load_model(self):
        """Load and cache the pretrained model for ``self.feature``."""
        feature = self.feature
        input_features = 4 if feature in _INTRABASE_FEATURES else 16
        model = DNAModel(
            input_features=input_features,
            filter_size=64,
            mp_layers=7,
            mp_steps=1,
            base_features=1,
            constraints=True,
            selflayer=True,
            gate=True,
            dual_weights=True,
            bn_layer=True,
            gru_layer=True,
        )
        weights_path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=f"{feature}.pt",
        )
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        self._model = model

    @torch.no_grad()
    def _predict_one(self, seq):
        """Predict shape values for a single DNA sequence string."""
        if self._model is None:
            self._load_model()
        model = self._model
        feature = self.feature
        layer = self.layer

        padded = "NN" + seq + "NN"
        rev = "".join(_REV_COMPLEMENT[b] for b in reversed(padded))

        if feature in _INTERBASE_FEATURES:

            def encode(s):
                return np.array([self._di[(s[i], s[i + 1])] for i in range(len(s) - 1)])
        else:

            def encode(s):
                return np.array([self._mono[b] for b in s])

        x_fwd = torch.tensor(encode(padded), dtype=torch.float32)
        x_rev = torch.tensor(encode(rev), dtype=torch.float32)

        pred_fwd = model(x_fwd).numpy()
        pred_rev = model(x_rev).numpy()

        params = self._params[feature]
        pred_fwd = _rescale(pred_fwd, params)
        pred_rev = _rescale(pred_rev, params)

        if feature in _X_AXIS_FEATURES:
            pred_rev = -pred_rev

        predictions = (pred_fwd + pred_rev[::-1]) / 2
        predictions = predictions.T[layer]
        return predictions[2:-2]

    def _transform(self, X):
        """Transform DNA sequences into shape feature rows.

        Parameters
        ----------
        X : pd.DataFrame
            Univariate frame; the first column holds DNA sequence strings.

        Returns
        -------
        pd.DataFrame
            One row per input sequence. Columns are positional shape
            values, right-padded with ``NaN`` to the longest prediction
            in the batch.
        """
        sequences = [_as_sequence_str(v) for v in X.iloc[:, 0].tolist()]
        preds = [
            np.asarray(self._predict_one(seq), dtype=np.float64) for seq in sequences
        ]

        max_len = max((len(p) for p in preds), default=0)
        padded = np.full((len(preds), max_len), np.nan, dtype=np.float64)
        for i, pred in enumerate(preds):
            padded[i, : len(pred)] = pred

        columns = [f"pos_{i}" for i in range(max_len)]
        return pd.DataFrame(padded, index=X.index, columns=columns)
