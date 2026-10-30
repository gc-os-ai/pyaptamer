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
    orientations. The two predictions are averaged. Sequences of the same
    length are scored together in batches.

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
    batch_size : int, default=256
        Maximum number of sequences of the same length scored in one pass
        through the model. Each sequence is scored on both strands, so a pass
        holds up to ``2 * batch_size`` rows. Lower it to reduce memory use for
        long sequences. It does not change the predictions.

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

    def __init__(self, feature="MGW", layer=4, batch_size=256):
        self.feature = feature
        self.layer = layer
        self.batch_size = batch_size
        super().__init__()

        if self.feature not in FEATURES:
            raise ValueError(f"Unknown feature. Must be one of {sorted(FEATURES)}.")
        if not 0 <= self.layer <= 7:
            raise ValueError(f"layer must be between 0 and 7, got {self.layer}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}.")

        self._model = None

    @classmethod
    def get_test_params(cls):
        """Return parameter sets for the shared transformer tests."""
        return [{}, {"feature": "Roll", "layer": 2, "batch_size": 2}]

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
    def _predict_batch(self, seqs):
        """Predict shape values for DNA sequence strings of one length.

        As in the original deepDNAshape, each sequence is flanked with two ``N``
        on each side, and both strands are predicted, all in one pass through
        the model. The reverse-strand prediction is flipped back to forward
        order, negated for features with ``flip_rev``, and averaged with the
        forward one. The flank positions are then dropped and the column for
        ``layer`` is returned.

        Parameters
        ----------
        seqs : list of str
            DNA sequences, all of the same length.

        Returns
        -------
        np.ndarray
            Array of shape (``len(seqs)``, number of positions).
        """
        if self._model is None:
            self._load_model()
        params = FEATURES[self.feature]

        complement = str.maketrans("ACGTN", "TGCAN")
        forward = ["NN" + seq + "NN" for seq in seqs]
        reverse = [seq[::-1].translate(complement) for seq in forward]
        batch = torch.stack(
            [_encode(seq, params.interbase) for seq in forward + reverse]
        )

        pred = self._model(batch)[..., self.layer].numpy()
        pred = pred * params.scale + params.offset
        pred_fwd, pred_rev = pred[: len(seqs)], pred[len(seqs) :]
        if params.flip_rev:
            pred_rev = -pred_rev

        predictions = (pred_fwd + pred_rev[:, ::-1]) / 2
        return predictions[:, 2:-2]

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

        seqs = X.iloc[:, 0].tolist()
        by_length = {}
        for i, seq in enumerate(seqs):
            by_length.setdefault(len(seq), []).append(i)

        preds = [None] * len(seqs)
        for rows in by_length.values():
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start : start + self.batch_size]
                values = self._predict_batch([seqs[i] for i in chunk])
                for i, row in zip(chunk, values, strict=True):
                    preds[i] = row
        return pd.DataFrame(preds, index=X.index, dtype=np.float64).add_prefix("pos_")
