"""Integration test: AptaNetPipeline with RNA aptamer sequences — Issue #696.

Verifies that the full pipeline handles RNA sequences (containing 'U')
without silent data loss, using the exact example from the user guide.
"""

import numpy as np
import pytest

from pyaptamer.aptanet import AptaNetPipeline


# RNA aptamer sequences from docs/source/user_guide/aptanet.md
RNA_APTAMERS = [
    "GGGAGGACGAAGACGACUCGAGACAGGCUAGGGAGGGA",
    "AAGCGUCGGAUCUACACGUGCGAUAGCUCAGUACGCGGU",
    "CGGUAUCGAGUACAGGAGUCCGACGGAUAGUCCGGAGC",
]
PROTEIN = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


class TestAptaNetPipelineRNA:
    """Integration tests for AptaNetPipeline with RNA sequences."""

    def test_pipeline_fit_predict_rna(self):
        """Pipeline should fit and predict on RNA aptamer sequences."""
        X = [(a, PROTEIN) for a in RNA_APTAMERS] * 10
        y = np.array([0, 1, 0] * 10, dtype=np.float32)

        pipe = AptaNetPipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X[:3])

        assert preds.shape == (3,)
        assert set(preds).issubset({0, 1})

    def test_pipeline_predict_proba_rna(self):
        """Pipeline should return valid probabilities for RNA sequences."""
        X = [(a, PROTEIN) for a in RNA_APTAMERS] * 10
        y = np.array([0, 1, 0] * 10, dtype=np.float32)

        pipe = AptaNetPipeline()
        pipe.fit(X, y)
        proba = pipe.predict_proba(X[:3])

        assert proba.shape == (3, 2)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_pipeline_with_explicit_alphabet(self):
        """Pipeline should accept an explicit alphabet parameter."""
        X = [(a, PROTEIN) for a in RNA_APTAMERS] * 10
        y = np.array([0, 1, 0] * 10, dtype=np.float32)

        pipe = AptaNetPipeline(alphabet="ACGU")
        pipe.fit(X, y)
        preds = pipe.predict(X[:3])

        assert preds.shape == (3,)
