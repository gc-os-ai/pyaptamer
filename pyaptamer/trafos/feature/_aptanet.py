"""AptaNet feature extraction transformations."""

__author__ = ["NandiniDhanrale"]
__all__ = [
    "AptaNetKmerTransformer",
    "AptaNetPSeAACTransformer",
    "AptaNetPairTransformer",
]

from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.utils import TransformerTags

from pyaptamer.pseaac import AptaNetPSeAAC
from pyaptamer.trafos.base import BaseTransform


def _aptanet_kmers(k: int) -> list[str]:
    """Generate all AptaNet DNA k-mers from length 1 to ``k``."""
    dna_bases = list("ACGT")
    all_kmers = []
    for i in range(1, k + 1):
        all_kmers.extend(["".join(p) for p in product(dna_bases, repeat=i)])
    return all_kmers


def _generate_kmer_vec(aptamer_sequence: str, k: int) -> np.ndarray:
    """Generate a normalized AptaNet k-mer frequency vector for one sequence."""
    aptamer_sequence = aptamer_sequence.replace("U", "T")
    all_kmers = _aptanet_kmers(k)
    kmer_counts = dict.fromkeys(all_kmers, 0)

    for i in range(len(aptamer_sequence)):
        for j in range(1, k + 1):
            if i + j <= len(aptamer_sequence):
                kmer = aptamer_sequence[i : i + j]
                if kmer in kmer_counts:
                    kmer_counts[kmer] += 1

    total_kmers = sum(kmer_counts.values())
    return np.array(
        [
            kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
            for kmer in all_kmers
        ]
    )


class _SklearnTransformerTagsMixin:
    """Expose sklearn transformer tags for sklearn Pipeline compatibility."""

    def __sklearn_tags__(self):
        tags = SklearnBaseEstimator.__sklearn_tags__(self)
        tags.transformer_tags = TransformerTags()
        return tags


class AptaNetKmerTransformer(_SklearnTransformerTagsMixin, BaseTransform):
    """Transform aptamer sequences into AptaNet k-mer frequency features.

    RNA bases are mapped to the AptaNet DNA convention by normalizing 'U' to 'T'.
    """

    _tags = {
        "authors": ["NandiniDhanrale"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, k: int = 4):
        self.k = k
        super().__init__()

    def _transform(self, X):
        """Transform a one-column DataFrame of aptamer sequences."""
        sequences = X.values[:, 0].tolist()
        features = np.vstack(
            [_generate_kmer_vec(str(seq), self.k) for seq in sequences]
        )
        columns = [f"kmer_{kmer}" for kmer in _aptanet_kmers(self.k)]
        return pd.DataFrame(features, index=X.index, columns=columns)


class AptaNetPSeAACTransformer(_SklearnTransformerTagsMixin, BaseTransform):
    """Transform protein sequences into AptaNet PSeAAC features."""

    _tags = {
        "authors": ["NandiniDhanrale"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, lambda_val: int = 30, weight: float = 0.05):
        self.lambda_val = lambda_val
        self.weight = weight
        super().__init__()

    def _transform(self, X):
        """Transform a one-column DataFrame of protein sequences."""
        pseaac = AptaNetPSeAAC(lambda_val=self.lambda_val, weight=self.weight)
        sequences = X.values[:, 0].tolist()
        features = np.vstack([pseaac.transform(str(seq)) for seq in sequences])
        columns = [f"pseaac_{i}" for i in range(features.shape[1])]
        return pd.DataFrame(features, index=X.index, columns=columns)


class AptaNetPairTransformer(_SklearnTransformerTagsMixin, BaseTransform):
    """Transform aptamer-protein sequence pairs into AptaNet features."""

    _tags = {
        "authors": ["NandiniDhanrale"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        k: int = 4,
        lambda_val: int = 30,
        weight: float = 0.05,
        aptamer_col: str = "aptamer",
        protein_col: str = "protein",
    ):
        self.k = k
        self.lambda_val = lambda_val
        self.weight = weight
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        super().__init__()

    def _check_X_y(self, X, y):  # noqa: N802
        """Check and coerce paired sequence inputs."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[self.aptamer_col, self.protein_col])
        return super()._check_X_y(X, y)

    def _check_X(self, X):  # noqa: N802
        """Check and coerce paired sequence inputs."""
        X, _ = self._check_X_y(X, None)
        return X

    def _transform(self, X):
        """Transform a DataFrame with aptamer and protein sequence columns."""
        aptamer_col = self.aptamer_col
        protein_col = self.protein_col
        if aptamer_col not in X.columns or protein_col not in X.columns:
            aptamer_col, protein_col = X.columns[:2]

        kmer = AptaNetKmerTransformer(k=self.k).fit_transform(X[[aptamer_col]])
        pseaac = AptaNetPSeAACTransformer(
            lambda_val=self.lambda_val, weight=self.weight
        ).fit_transform(X[[protein_col]])

        return pd.concat([kmer, pseaac], axis=1)
