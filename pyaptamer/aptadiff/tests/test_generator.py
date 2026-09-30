"""Test the AptaDiffGenerator estimator."""

__author__ = ["aditi-dsi"]

import copy
import pickle
from collections.abc import Callable
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptadiff import AptaDiffGenerator
from pyaptamer.aptadiff.layers._libs._linear_attention_transformer import (
    LinearAttentionTransformer,
)

NUM_CLASSES = 4
SEQ_LEN = 8
N_LATENT = 2
N_SAMPLES = 20
TEST_PARAMS = {
    "num_timesteps": 4,
    "dim": 32,
    "depth": 1,
    "max_epochs": 1,
    "random_state": 0,
}


def _tokens(y: np.ndarray) -> np.ndarray:
    """Return the class index at each position of flat one-hot sequences."""
    return y.reshape(len(y), NUM_CLASSES, SEQ_LEN).argmax(axis=1)


def _expected_failed_checks(estimator: AptaDiffGenerator) -> dict[str, str]:
    """Return the sklearn checks that fail because their y is not one-hot."""
    return dict.fromkeys(
        [
            "check_fit_score_takes_y",
            "check_estimators_overwrite_params",
            "check_dont_overwrite_parameters",
            "check_estimators_fit_returns_self",
            "check_readonly_memmap_input",
            "check_n_features_in_after_fitting",
            "check_positive_only_tag_during_fit",
        ],
        "sklearn passes a 1D y, not flat one-hot sequences",
    )


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    """Return latent vectors and flat one-hot sequences in the fit layout."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, NUM_CLASSES, size=(N_SAMPLES, SEQ_LEN))
    y = np.eye(NUM_CLASSES, dtype=np.float32)[tokens]
    y = y.transpose(0, 2, 1).reshape(N_SAMPLES, -1)
    X = rng.normal(size=(N_SAMPLES, N_LATENT)).astype(np.float32)

    return X, y


@pytest.fixture(scope="module")
def fitted(data: tuple[np.ndarray, np.ndarray]) -> AptaDiffGenerator:
    """Return a small AptaDiffGenerator fitted on data."""
    X, y = data
    return AptaDiffGenerator(**TEST_PARAMS).fit(X, y)


class TestAptaDiffGenerator:
    """Test AptaDiffGenerator fitting and generation."""

    @pytest.mark.parametrize(
        "params, block_type",
        [
            ({}, nn.TransformerEncoderLayer),
            (
                {
                    "transformer_type": "linear",
                    "n_local_attn_heads": 1,
                    "local_attn_window_size": 2,
                },
                LinearAttentionTransformer,
            ),
        ],
    )
    def test_fit_predict_end_to_end(
        self,
        data: tuple[np.ndarray, np.ndarray],
        params: dict[str, Any],
        block_type: type[nn.Module],
    ) -> None:
        """Check fit builds the configured model and predict returns one-hot."""
        X, y = data
        generator = AptaDiffGenerator(**TEST_PARAMS, **params)
        training_device = L.Trainer(
            accelerator="auto", devices=1, logger=False, enable_progress_bar=False
        ).strategy.root_device

        assert generator.fit(X, y) is generator
        assert generator.seq_len_ == SEQ_LEN
        assert generator.n_features_in_ == N_LATENT

        diffusion = generator.diffusion_
        block = diffusion.denoise_fn.transformer.transformer_blocks[0][0]
        assert type(block) is block_type
        assert not diffusion.training
        assert next(diffusion.parameters()).device.type == training_device.type

        out = generator.predict(X[:5])

        assert out.dtype == np.float32
        assert out.shape == (5, NUM_CLASSES * SEQ_LEN)
        onehot = out.reshape(5, NUM_CLASSES, SEQ_LEN)
        assert np.isin(onehot, (0.0, 1.0)).all()
        assert (onehot.sum(axis=1) == 1).all()

    def test_predict_is_unaffected_by_global_seed(
        self, fitted: AptaDiffGenerator, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Check predict's output is not affected by the global seed."""
        X, _ = data

        torch.manual_seed(1)
        first = fitted.predict(X)
        torch.manual_seed(2)
        second = fitted.predict(X)

        np.testing.assert_array_equal(first, second)

    def test_fit_and_predict_leave_global_rng_untouched(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Check fit and predict with random_state doesn;t affect the global
        RNG state."""
        X, y = data
        torch_before = torch.get_rng_state()
        numpy_before = np.random.get_state()[1].copy()

        AptaDiffGenerator(**TEST_PARAMS).fit(X, y).predict(X)

        assert torch.equal(torch_before, torch.get_rng_state())
        np.testing.assert_array_equal(numpy_before, np.random.get_state()[1])

    @pytest.mark.parametrize("to_cpu", [False, True])
    def test_pickle_roundtrip(
        self,
        fitted: AptaDiffGenerator,
        data: tuple[np.ndarray, np.ndarray],
        to_cpu: bool,
    ) -> None:
        """Check an unpickled generator predicts the same sequences."""
        X, _ = data
        generator = copy.deepcopy(fitted)
        if to_cpu:
            generator.diffusion_.cpu()
        expected = generator.predict(X)

        restored = pickle.loads(pickle.dumps(generator))

        np.testing.assert_array_equal(restored.predict(X), expected)
        if to_cpu:
            assert next(restored.diffusion_.parameters()).device.type == "cpu"


class TestAptaDiffGeneratorInputs:
    """Test AptaDiffGenerator input validation and conversion."""

    @pytest.mark.parametrize(
        "make_invalid_y, match",
        [
            pytest.param(
                lambda y: y[:, :-1],
                "not a multiple of num_classes",
                id="width-not-multiple",
            ),
            pytest.param(
                lambda y: _tokens(y).astype(np.float32), "one-hot", id="raw-tokens"
            ),
            pytest.param(
                lambda y: np.eye(NUM_CLASSES, dtype=np.float32)[_tokens(y)].reshape(
                    len(y), -1
                ),
                "one-hot",
                id="position-major",
            ),
            pytest.param(lambda y: y[:, 0], "must be a 2D array", id="1d"),
        ],
    )
    def test_fit_rejects_invalid_y(
        self,
        data: tuple[np.ndarray, np.ndarray],
        make_invalid_y: Callable[[np.ndarray], np.ndarray],
        match: str,
    ) -> None:
        """Check y that is not flat one-hot raises a ValueError."""
        X, y = data

        with pytest.raises(ValueError, match=match):
            AptaDiffGenerator(**TEST_PARAMS).fit(X, make_invalid_y(y))

    def test_predict_rejects_wrong_latent_width(
        self, fitted: AptaDiffGenerator
    ) -> None:
        """Check latent vectors of a different width than in fit raise a ValueError."""
        X = np.zeros((3, N_LATENT + 1), dtype=np.float32)

        with pytest.raises(ValueError, match=f"expecting {N_LATENT} features"):
            fitted.predict(X)

    def test_fit_accepts_float64_and_dataframe_inputs(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Check float64 arrays and a y DataFrame are converted to float32."""
        X, y = data
        X = X.astype(np.float64)
        y = pd.DataFrame(y.astype(np.float64))

        generator = AptaDiffGenerator(**TEST_PARAMS).fit(X, y)
        out = generator.predict(X)

        assert out.dtype == np.float32
        assert out.shape == (N_SAMPLES, NUM_CLASSES * SEQ_LEN)


class TestAptaDiffGeneratorSklearn:
    """Test AptaDiffGenerator against scikit-learn's estimator checks."""

    @parametrize_with_checks(
        [AptaDiffGenerator(**TEST_PARAMS)],
        legacy=False,
        expected_failed_checks=_expected_failed_checks,
    )
    def test_sklearn_compatible_estimator(
        self, estimator: AptaDiffGenerator, check: Callable[[Any], None]
    ) -> None:
        """Run scikit-learn's API checks on AptaDiffGenerator."""
        check(estimator)
