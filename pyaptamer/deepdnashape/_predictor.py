"""DeepDNAshape predictor for DNA shape feature prediction."""

__author__ = ["prashantpandeygit"]
__all__ = ["Predictor"]

import itertools
import json
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download

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
    bits = len(bases)

    mono = {}
    for i, base in enumerate(bases):
        vec = np.zeros(bits)
        vec[i] = 1
        mono[base] = vec
    mono["N"] = np.ones(bits, dtype=float) / bits

    di = {("N", "N"): np.ones(bits * bits, dtype=float) / bits / bits}
    for i, pair in enumerate(itertools.product(bases, repeat=2)):
        vec = np.zeros(bits * bits)
        vec[i] = 1
        di[pair] = vec

    for bp in bases:
        di[(bp, "N")] = np.zeros(bits * bits)
        di[("N", bp)] = np.zeros(bits * bits)
        for bp2 in bases:
            di[(bp, "N")] += di[(bp, bp2)]
            di[("N", bp)] += di[(bp2, bp)]
        di[(bp, "N")] /= bits
        di[("N", bp)] /= bits

    return mono, di


def _build_graph(x):
    """Construct edge indices for a linear chain graph with self-loops.

    Each of the N nodes (sequence positions) is connected to its
    immediate predecessor and successor. The first and last nodes
    receive additional self-loop edges so that every node has the
    same number of incoming edges.

    Parameters
    ----------
    x : torch.Tensor of shape (N, C)
        Node feature matrix (passed through unchanged).

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
        (x, pairs_prev, pairs_next) where pairs_prev and
        pairs_next are (E, 2) edge index tensors consumed
        by MessagePassingConv.
    """
    k = x.shape[0]
    rng = torch.arange(k - 1, dtype=torch.long)

    edges = torch.repeat_interleave(rng, 4).reshape(-1, 4)
    pad = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    edges = edges + pad

    edges = edges.reshape(-1, 2)
    self_start = torch.zeros(1, 2, dtype=torch.long)
    self_end = torch.full((1, 2), k - 1, dtype=torch.long)
    edges = torch.cat([self_start, edges, self_end], dim=0)

    edges = edges.reshape(-1, 4)
    return x, edges[:, :2], edges[:, 2:]


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


class Predictor:
    """Predictor for DNA structural shape features from nucleotide sequence. [1]_

    This class wraps the DNAModel graph neural network to predict
    geometric and conformational properties of DNA (e.g. Minor Groove
    Width, Roll, Helical Twist) from a raw sequence string.

    For each prediction the sequence is one-hot encoded, padded with
    two N bases on each side, and run through the model in both
    forward and reverse-complement orientations. The two sets of
    predictions are averaged to produce the final result.

    Original implementation: https://github.com/JinsenLi/deepDNAshape
    License: BSD-3-Clause

    Examples
    --------
    >>> from pyaptamer.deepdnashape import Predictor
    >>> pred = Predictor()
    >>> scores = pred.predict("MGW", "AAGGTAGT")
    """

    def __init__(self):
        self._mono, self._di = _get_bases_mapping()
        with open(_PARAMS_PATH) as f:
            self._params = json.load(f)
        self._models = {}

    def _load_model(self, feature):
        """Load and cache pretrained model for the given feature."""
        input_features = 4 if feature in _INTRABASE_FEATURES else 16
        model = DNAModel(
            input_features=input_features,
            filter_size=64,
            mp_layers=7,
            mp_steps=1,
            base_features=1,
            constraints=True,
            multiply="add",
            selflayer=True,
            bn_layer=True,
            gru_layer=True,
        )
        weights_path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=f"{feature}.pt",
        )
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        self._models[feature] = model

    @torch.no_grad()
    def predict(self, feature, seq, layer=4):
        """Predict DNA shape values for a single sequence.

        Parameters
        ----------
        feature : str
            Name of the structural property to predict, e.g.
            "MGW" (Minor Groove Width), "Roll",
            "HelT" (Helical Twist). See _ALL_FEATURES
            for the full list.
        seq : str
            DNA sequence composed of A, T, C,
            G, and optionally N (unknown base).
        layer : int, optional
            Which message-passing depth to read the
            prediction from (0 = initial convolution only,
            7 = deepest layer). Default is 4.

        Returns
        -------
        np.ndarray
            Predicted shape values, one per position in the input sequence.
        """
        if feature not in _ALL_FEATURES:
            raise ValueError(
                f"Unknown feature. Must be one of {sorted(_ALL_FEATURES)}."
            )
        if not 0 <= layer <= 7:
            raise ValueError(f"layer must be between 0 and 7, got {layer}.")

        if feature not in self._models:
            self._load_model(feature)
        model = self._models[feature]

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

        x_fwd, prev_fwd, next_fwd = _build_graph(x_fwd)
        x_rev, prev_rev, next_rev = _build_graph(x_rev)

        pred_fwd = model(x_fwd, prev_fwd, next_fwd).numpy()
        pred_rev = model(x_rev, prev_rev, next_rev).numpy()

        params = self._params[feature]
        pred_fwd = _rescale(pred_fwd, params)
        pred_rev = _rescale(pred_rev, params)

        if feature in _X_AXIS_FEATURES:
            pred_rev = -pred_rev

        predictions = (pred_fwd + pred_rev[::-1]) / 2
        predictions = predictions.T[layer]
        return predictions[2:-2]
