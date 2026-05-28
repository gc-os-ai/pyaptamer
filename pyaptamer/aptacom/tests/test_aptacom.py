import numpy as np
import pytest

from pyaptamer.aptacom import AptaComPipeline
from pyaptamer.aptacom._aptacom_utils import (
    aptacom_dna_features,
    aptacom_dna_sequences_to_features,
)


class TestAptaComDNAFeatures:
    def test_aptacom_dna_features_shape(self):
        seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
        feat = aptacom_dna_features(seq)
        assert isinstance(feat, np.ndarray)
        assert feat.dtype == np.float32
        assert feat.ndim == 1
        assert feat.shape[0] > 0

    def test_aptacom_dna_features_deterministic(self):
        seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
        feat1 = aptacom_dna_features(seq)
        feat2 = aptacom_dna_features(seq)
        np.testing.assert_array_equal(feat1, feat2)

    def test_aptacom_dna_features_u_conversion(self):
        seq_t = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
        seq_u = seq_t.replace("T", "U")
        feat_t = aptacom_dna_features(seq_t)
        feat_u = aptacom_dna_features(seq_u)
        np.testing.assert_array_equal(feat_t, feat_u)


class TestAptaComSequencesToFeatures:
    def test_aptacom_sequences_to_features_shape(self):
        seqs = [
            "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC",
            "GGCGTAGCGATAGATAGCTAGCTAGCTAGCTAGC",
            "ATATAT",
        ]
        X = aptacom_dna_sequences_to_features(seqs)
        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float32
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_aptacom_sequences_to_features_consistent_cols(self):
        seqs1 = ["AGCTTAGCGTAC"]
        seqs2 = ["AGCTTAGCGTAC", "GGCGTAGCGAT"]
        X1 = aptacom_dna_sequences_to_features(seqs1)
        X2 = aptacom_dna_sequences_to_features(seqs2)
        assert X1.shape[1] == X2.shape[1]


class TestAptaComPipeline:
    def test_pipeline_fit_and_predict(self):
        seqs = [
            "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC",
            "GGCGTAGCGATAGATAGCTAGCTAGCTAGCTAGC",
            "ATATAT",
        ] * 10
        y = np.array([0, 1, 0] * 10)
        pipe = AptaComPipeline(random_state=42)
        pipe.fit(seqs, y)
        preds = pipe.predict(seqs)
        assert preds.shape == (30,)
        assert np.all((preds == 0) | (preds == 1))

    def test_pipeline_fit_and_predict_proba(self):
        seqs = ["AGCTTAGCGTAC"] * 20 + ["GGCGTAGCGAT"] * 20
        y = np.array([0] * 20 + [1] * 20)
        pipe = AptaComPipeline(random_state=42)
        pipe.fit(seqs, y)
        proba = pipe.predict_proba(seqs)
        assert proba.shape == (40, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_pipeline_returns_self_on_fit(self):
        seqs = ["AGCTTAGCGTAC"] * 10
        y = np.array([0, 1] * 5)
        pipe = AptaComPipeline()
        result = pipe.fit(seqs, y)
        assert result is pipe

    def test_pipeline_not_fitted_raises(self):
        pipe = AptaComPipeline()
        seqs = ["AGCTTAGCGTAC"]
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            pipe.predict(seqs)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_pipeline_different_k(self, k):
        seqs = ["AGCTTAGCGTAC"] * 10
        y = np.array([0, 1] * 5)
        pipe = AptaComPipeline(k=k, random_state=42)
        pipe.fit(seqs, y)
        preds = pipe.predict(seqs)
        assert preds.shape == (10,)
