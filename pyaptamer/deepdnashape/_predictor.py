"""DeepDNAShape transform for DNA shape feature prediction."""

__author__ = ["prashantpandeygit", "Alleny244"]
__all__ = ["DeepDNAShape"]

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from pyaptamer.deepdnashape._model import DNAModel
from pyaptamer.deepdnashape._params import BASE_VECTORS, FEATURES
from pyaptamer.transformations.base import BaseTransform


def _encode(seq, interbase):
    """Encode a DNA sequence as the model input.

    Each base is a one-hot vector in the order T, G, C, A, the order the
    pretrained weights expect. ``N`` is the average of the four bases.

    Intrabase features use one vector per base, shape (``len(seq)``, 4).
    Interbase features use one vector per step between two bases: the outer
    product of the two base vectors, flattened to a one-hot over the 16 base
    pairs, shape (``len(seq) - 1``, 16). A step with ``N`` is the average of
    the pairs it could be.

    Parameters
    ----------
    seq : str
        DNA sequence over A, C, G, T and N.
    interbase : bool
        Whether to encode steps between bases instead of single bases.

    Returns
    -------
    torch.Tensor
        The encoded sequence.
    """
    bases = torch.tensor([BASE_VECTORS[b] for b in seq])
    if not interbase:
        return bases
    return (bases[:-1, :, None] * bases[1:, None, :]).flatten(start_dim=1)


class DeepDNAShape(BaseTransform):
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
    >>> from pyaptamer.deepdnashape import DeepDNAShape
    >>> X = pd.DataFrame({"seq": ["AAGGTAGT"]})
    >>> mgw = DeepDNAShape(feature="MGW").fit_transform(X)
    >>> roll = DeepDNAShape(feature="Roll").fit_transform(X)
    """

    _tags = {
        "authors": ["prashantpandeygit", "Alleny244"],
        "maintainers": ["siddharth7113"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
        "capability:y": False,
    }

    def __init__(self, feature="MGW", layer=4):
        self.feature = feature
        self.layer = layer
        super().__init__()

        if self.feature not in FEATURES:
            raise ValueError(f"Unknown feature. Must be one of {sorted(FEATURES)}.")
        if not 0 <= self.layer <= 7:
            raise ValueError(f"layer must be between 0 and 7, got {self.layer}.")

        self._model = None

    def _load_model(self):
        """Load and cache the pretrained model for ``self.feature``."""
        model = DNAModel(
            input_features=16 if FEATURES[self.feature].interbase else 4,
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
            repo_id="ecospecs/deepdnashape", filename=f"{self.feature}.pt"
        )
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        self._model = model

    @torch.no_grad()
    def _predict_one(self, seq):
        """Predict shape values for a single DNA sequence string.

        As in the original deepDNAshape, the sequence is flanked with two ``N``
        on each side, and both strands are predicted. The reverse-strand
        prediction is flipped back to forward order, negated for features with
        ``flip_rev``, and averaged with the forward one. The flank positions are
        then dropped and the column for ``layer`` is returned.
        """
        if self._model is None:
            self._load_model()
        params = FEATURES[self.feature]

        forward = "NN" + seq + "NN"
        reverse = forward[::-1].translate(str.maketrans("ACGTN", "TGCAN"))

        pred_fwd = self._model(_encode(forward, params.interbase)).numpy()
        pred_rev = self._model(_encode(reverse, params.interbase)).numpy()
        pred_fwd = pred_fwd * params.scale + params.offset
        pred_rev = pred_rev * params.scale + params.offset
        if params.flip_rev:
            pred_rev = -pred_rev

        predictions = (pred_fwd + pred_rev[::-1]) / 2
        return predictions[2:-2, self.layer]

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

        Raises
        ------
        TypeError
            If the first column of X does not hold one sequence string per row.
        ValueError
            If a sequence has a base other than A, C, G, T and N.
        """
        kind = pd.api.types.infer_dtype(X.iloc[:, 0], skipna=True)
        if kind not in ("string", "empty"):
            raise TypeError(
                f"{type(self).__name__} needs one sequence string per row. A "
                'MoleculeLoader built from a file with tiling="bag" holds several '
                'sequences per row. Build it with tiling="samples".'
            )
        for seq in X.iloc[:, 0]:
            invalid = set(seq) - set("ACGTN")
            if invalid:
                raise ValueError(
                    f"Invalid bases {sorted(invalid)} in {seq!r}. "
                    "Allowed bases are A, C, G, T, N."
                )

        preds = [self._predict_one(seq) for seq in X.iloc[:, 0]]
        return pd.DataFrame(preds, index=X.index, dtype=np.float64).add_prefix("pos_")
