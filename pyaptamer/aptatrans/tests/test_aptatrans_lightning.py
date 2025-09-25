"""Test suite for AptaTrans' wrapper to lightning."""

__author__ = ["nennomp"]

import pytest
import torch
import torch.nn as nn

from pyaptamer.aptatrans import AptaTransLightning


class TestAptaTransLightning:
    """Tests for the AptaTransLightning() class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock AptaTrans model for testing purposes."""

        class MockAptaTrans(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Parameter(torch.zeros(1))

            def forward(self, x_apta, x_prot):
                batch_size = x_apta.shape[0]
                return torch.rand(batch_size, 1)

        return MockAptaTrans()

    @pytest.fixture
    def lightning_model(self, mock_model):
        """Create AptaTransLightning instance with default parameters."""
        return AptaTransLightning(mock_model)

    def test_init_default_params(self, mock_model):
        """Check that default parameters are set correctly during initialization."""
        lightning_model = AptaTransLightning(mock_model)

        assert lightning_model.model is mock_model
        assert lightning_model.lr == 1e-5
        assert lightning_model.weight_decay == 1e-5
        assert lightning_model.betas == (0.9, 0.999)

    def test_init_custom_params(self, mock_model):
        """Check that custom parameters are set correctly during initialization."""
        lr = 1e-4
        weight_decay = 1e-3
        betas = (0.8, 0.99)

        lightning_model = AptaTransLightning(
            mock_model, lr=lr, weight_decay=weight_decay, betas=betas
        )

        assert lightning_model.lr == lr
        assert lightning_model.weight_decay == weight_decay
        assert lightning_model.betas == betas

    @pytest.mark.parametrize(
        "batch_size, seq_len",
        [(4, 50), (8, 100), (2, 75)],
    )
    def test_training_step(self, lightning_model, batch_size, seq_len):
        """Check training_step computes loss correctly."""
        # create dummy batch
        x_apta = torch.randint(0, 4, (batch_size, seq_len))
        x_prot = torch.randint(0, 20, (batch_size, seq_len))
        y = torch.randint(0, 2, (batch_size, 1)).float()
        batch = (x_apta, x_prot, y)

        loss = lightning_model.training_step(batch, batch_idx=0)

        # check that loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # bce loss should be non-negative

    def test_configure_optimizers(self, lightning_model):
        """Check that optimizer is configured correctly."""
        optimizer = lightning_model.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == lightning_model.lr
        assert optimizer.defaults["weight_decay"] == lightning_model.weight_decay
        assert optimizer.defaults["betas"] == lightning_model.betas
