__all__ = ["pairs_to_features"]

import numpy as np
import pandas as pd

from pyaptamer.utils._pseaac_utils import AMINO_ACIDS
from pyaptamer.utils._rna import dna2rna


def _normalized_char_counts(sequence: str, alphabet: list[str]) -> np.ndarray:
    """Convert sequence into normalized character frequencies."""
    if not sequence:
        return np.zeros(len(alphabet), dtype=np.float32)

    idx = {char: i for i, char in enumerate(alphabet)}
    counts = np.zeros(len(alphabet), dtype=np.float32)
    total = 0

    for char in sequence:
        i = idx.get(char)
        if i is not None:
            counts[i] += 1.0
            total += 1

    if total == 0:
        return counts

    return counts / float(total)


def pairs_to_features(pairs) -> np.ndarray:
    """Encode aptamer-target pairs into numerical features.

    This is a minimal iCTF-style placeholder encoder used for AptaMCTS integration.
    It produces compact composition-based features from aptamer and target sequences.

    Parameters
    ----------
    pairs : list[tuple[str, str]] or pandas.DataFrame
        Sequence pairs ``(aptamer, target)`` or a DataFrame with ``aptamer`` and
        ``protein`` columns.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, 27)`` with dtype ``np.float32``.
    """
    if isinstance(pairs, pd.DataFrame):
        sequence_pairs = zip(pairs["aptamer"], pairs["protein"], strict=False)
    else:
        sequence_pairs = pairs

    features = []
    amino_acids = list(AMINO_ACIDS)
    aptamer_alphabet = ["A", "C", "G", "U", "N"]

    for aptamer, target in sequence_pairs:
        aptamer = dna2rna(str(aptamer).upper())
        target = str(target).upper()

        aptamer_freq = _normalized_char_counts(aptamer, aptamer_alphabet)
        target_freq = _normalized_char_counts(target, amino_acids)

        lengths = np.array([float(len(aptamer)), float(len(target))], dtype=np.float32)
        lengths /= max(float(max(len(aptamer), len(target))), 1.0)

        features.append(np.concatenate([aptamer_freq, target_freq, lengths]))

    if not features:
        return np.empty((0, 27), dtype=np.float32)

    return np.vstack(features).astype(np.float32, copy=False)
