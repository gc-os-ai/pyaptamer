"""Tests for AptaTransPipeline.fit() and predict_interactions() — issue #190."""

__author__ = ["Ishiezz"]

import numpy as np
import pandas as pd
import pytest
import torch

from pyaptamer.aptatrans import AptaTrans, AptaTransPipeline, EncoderPredictorConfig

# Shared fixtures


@pytest.fixture(scope="module")
def small_pipeline():
    """Minimal AptaTransPipeline for fast CPU testing."""
    device = torch.device("cpu")
    apta_cfg = EncoderPredictorConfig(126, 4, max_len=20)
    prot_cfg = EncoderPredictorConfig(1001, 4, max_len=20)
    model = AptaTrans(apta_cfg, prot_cfg)
    prot_words = {
        "DHR": 0.9,
        "HRN": 0.8,
        "RNE": 0.7,
        "NEN": 0.6,
        "ENI": 0.3,
        "NIA": 0.2,
        "IAI": 0.1,
        "AIQ": 0.05,
    }
    return AptaTransPipeline(device=device, model=model, prot_words=prot_words)


def _make_dataframe(n: int = 8) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a tiny DataFrame + binary labels for testing."""
    X = pd.DataFrame(
        {
            "aptamer": ["ACGUACGU"] * n,
            "protein": ["DHRNENIAIQ"] * n,
        }
    )
    y = np.array([1, 0] * (n // 2), dtype=np.float32)
    return X, y


# fit() tests


class TestAptaTransPipelineFit:
    def test_fit_returns_self(self, small_pipeline):
        """fit() must return self to allow method chaining."""
        X, y = _make_dataframe()
        result = small_pipeline.fit(X, y, max_epochs=1, batch_size=4, accelerator="cpu")
        assert result is small_pipeline

    def test_fit_sets_is_fitted(self, small_pipeline):
        """After fit(), is_fitted_ must be True."""
        X, y = _make_dataframe()
        small_pipeline.fit(X, y, max_epochs=1, batch_size=4, accelerator="cpu")
        assert small_pipeline.is_fitted_ is True

    def test_fit_accepts_list_of_tuples(self, small_pipeline):
        """fit() must accept list[tuple[str, str]] input via APIDataset.from_any."""
        X, y = _make_dataframe()
        pairs = list(zip(X["aptamer"], X["protein"], strict=False))
        small_pipeline.fit(pairs, y, max_epochs=1, batch_size=4, accelerator="cpu")
        assert small_pipeline.is_fitted_

    def test_fit_accepts_numpy_pair(self, small_pipeline):
        """fit() must accept (np.ndarray, np.ndarray) input."""
        X, y = _make_dataframe()
        apta = X["aptamer"].to_numpy()
        prot = X["protein"].to_numpy()
        small_pipeline.fit(
            (apta, prot), y, max_epochs=1, batch_size=4, accelerator="cpu"
        )
        assert small_pipeline.is_fitted_


# predict_interactions() tests


class TestAptaTransPipelinePredictInteractions:
    def test_predict_before_fit_raises(self, small_pipeline):
        """predict_interactions() before fit() must raise RuntimeError."""
        device = torch.device("cpu")
        apta_cfg = EncoderPredictorConfig(126, 4, max_len=20)
        prot_cfg = EncoderPredictorConfig(1001, 4, max_len=20)
        model = AptaTrans(apta_cfg, prot_cfg)
        fresh_pipeline = AptaTransPipeline(
            device=device, model=model, prot_words={"DHR": 0.9, "AIQ": 0.3}
        )
        X, _ = _make_dataframe()
        with pytest.raises(RuntimeError, match="not fitted yet"):
            fresh_pipeline.predict_interactions(X)

    def test_predict_output_shape(self, small_pipeline):
        """predict_interactions() must return array of shape (n_samples,)."""
        X, y = _make_dataframe(n=8)
        small_pipeline.fit(X, y, max_epochs=1, batch_size=4, accelerator="cpu")
        preds = small_pipeline.predict_interactions(X, batch_size=4, accelerator="cpu")
        assert preds.shape == (len(X),)

    def test_predict_output_range(self, small_pipeline):
        """predict_interactions() must return probabilities in [0, 1]."""
        X, y = _make_dataframe(n=8)
        small_pipeline.fit(X, y, max_epochs=1, batch_size=4, accelerator="cpu")
        preds = small_pipeline.predict_interactions(X, batch_size=4, accelerator="cpu")
        assert float(preds.min()) >= 0.0
        assert float(preds.max()) <= 1.0

    def test_predict_returns_numpy(self, small_pipeline):
        """predict_interactions() must return np.ndarray, not a Tensor."""
        X, y = _make_dataframe(n=8)
        small_pipeline.fit(X, y, max_epochs=1, batch_size=4, accelerator="cpu")
        preds = small_pipeline.predict_interactions(X, batch_size=4, accelerator="cpu")
        assert isinstance(preds, np.ndarray)

    def test_predict_output_is_1d(self, small_pipeline):
        """predict_interactions() must return a 1D array."""
        X, y = _make_dataframe(n=8)
        small_pipeline.fit(X, y, max_epochs=1, batch_size=4, accelerator="cpu")
        preds = small_pipeline.predict_interactions(X, batch_size=4, accelerator="cpu")
        assert preds.ndim == 1
