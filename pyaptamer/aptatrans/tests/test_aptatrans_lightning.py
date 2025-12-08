"""Test suite for AptaTrans' wrapper to lightning."""

__author__ = ["nennomp"]

import pytest
import torch
import torch.nn as nn

from pyaptamer.aptatrans import AptaTransEncoderLightning, AptaTransLightning


@pytest.fixture
def mock_model():
    """Create a mock AptaTrans model for testing purposes."""

    class MockAptaTrans(nn.Module):
        def __init__(self):
            super().__init__()
            # dummy parameters for testing purposes
            self.dummy_param = nn.Parameter(torch.zeros(10))

            # encoder components needed by configure_optimizers
            self.encoder_apta = nn.Linear(10, 10)
            self.encoder_prot = nn.Linear(10, 10)
            self.token_predictor_apta = nn.Linear(10, 10)
            self.token_predictor_prot = nn.Linear(10, 10)

        def forward_encoder(self, x, encoder_type):
            batch_size, seq_len = x[0].shape
            vocab_size = 125
            ss_classes = 8

            return (
                torch.randn(batch_size, seq_len, vocab_size),
                torch.randn(batch_size, seq_len, ss_classes),
            )

        def forward(self, x_apta, x_prot):
            batch_size = x_apta.shape[0]
            return torch.rand(batch_size, 1)

    return MockAptaTrans()


class TestAptaTransLightning:
    """Tests for the AptaTransLightning() class."""

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
        """Check `training_step` computes loss correctly with labels."""
        x_apta = torch.randint(0, 4, (batch_size, seq_len))
        x_prot = torch.randint(0, 20, (batch_size, seq_len))
        y = torch.randint(0, 2, (batch_size,)).float()
        batch = (x_apta, x_prot, y)

        loss = lightning_model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "batch_size, seq_len",
        [(4, 50), (8, 100), (2, 75)],
    )
    def test_test_step(self, lightning_model, batch_size, seq_len):
        """Check `test_step` computes loss correctly with labels."""
        x_apta = torch.randint(0, 4, (batch_size, seq_len))
        x_prot = torch.randint(0, 20, (batch_size, seq_len))
        y = torch.randint(0, 2, (batch_size,)).float()
        batch = (x_apta, x_prot, y)

        loss = lightning_model.test_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "batch_size, seq_len",
        [(4, 50), (8, 100), (2, 75)],
    )
    def test_predict_step(self, lightning_model, batch_size, seq_len):
        """Check `predict_step` returns binary predictions without labels."""
        x_apta = torch.randint(0, 4, (batch_size, seq_len))
        x_prot = torch.randint(0, 20, (batch_size, seq_len))
        batch = (x_apta, x_prot)

        predictions = lightning_model.predict_step(batch, batch_idx=0)

        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == (batch_size,)
        assert torch.all((predictions == 0) | (predictions == 1))

    def test_configure_optimizers(self, lightning_model):
        """Check that optimizer is configured correctly."""
        optimizer = lightning_model.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == lightning_model.lr
        assert optimizer.defaults["weight_decay"] == lightning_model.weight_decay
        assert optimizer.defaults["betas"] == lightning_model.betas


class TestAptaTransEncoderLightning:
    """Tests for the AptaTransEncoderLightning() class."""

    @pytest.fixture
    def lightning_model(self, mock_model):
        """Create AptaTransEncoderLightning instance with default parameters."""
        return AptaTransEncoderLightning(mock_model, encoder_type="apta")

    def test_init_default_params(self, mock_model):
        """Check that default parameters are set correctly during initialization."""
        lightning_model = AptaTransEncoderLightning(mock_model, encoder_type="apta")

        assert lightning_model.model is mock_model
        assert lightning_model.encoder_type == "apta"
        assert lightning_model.lr == 1e-4
        assert lightning_model.weight_decay == 1e-5
        assert lightning_model.betas == (0.9, 0.999)
        assert lightning_model.weight_mlm == 2.0
        assert lightning_model.weight_ssp == 1.0

    def test_init_custom_params(self, mock_model):
        """Check that custom parameters are set correctly during initialization."""
        lr = 1e-3
        weight_decay = 1e-4
        betas = (0.8, 0.99)
        weight_mlm = 3.0
        weight_ssp = 0.5

        lightning_model = AptaTransEncoderLightning(
            mock_model,
            encoder_type="prot",
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            weight_mlm=weight_mlm,
            weight_ssp=weight_ssp,
        )

        assert lightning_model.encoder_type == "prot"
        assert lightning_model.lr == lr
        assert lightning_model.weight_decay == weight_decay
        assert lightning_model.betas == betas
        assert lightning_model.weight_mlm == weight_mlm
        assert lightning_model.weight_ssp == weight_ssp

    @pytest.mark.parametrize(
        "batch_size, seq_len",
        [(4, 50), (8, 100), (2, 75)],
    )
    def test_training_step(self, lightning_model, batch_size, seq_len):
        """Check `training_step` computes loss correctly with labels."""
        x_mlm = torch.randint(0, 125, (batch_size, seq_len))
        x_ssp = torch.randint(0, 125, (batch_size, seq_len))
        y_mlm = torch.randint(0, 125, (batch_size, seq_len))
        y_ssp = torch.randint(0, 8, (batch_size, seq_len))
        batch = (x_mlm, x_ssp, y_mlm, y_ssp)

        loss = lightning_model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "batch_size, seq_len",
        [(4, 50), (8, 100), (2, 75)],
    )
    def test_test_step(self, lightning_model, batch_size, seq_len):
        """Check `test_step` computes loss correctly with labels."""
        x_mlm = torch.randint(0, 125, (batch_size, seq_len))
        x_ssp = torch.randint(0, 125, (batch_size, seq_len))
        y_mlm = torch.randint(0, 125, (batch_size, seq_len))
        y_ssp = torch.randint(0, 8, (batch_size, seq_len))
        batch = (x_mlm, x_ssp, y_mlm, y_ssp)

        loss = lightning_model.test_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "batch_size, seq_len",
        [(4, 50), (8, 100), (2, 75)],
    )
    def test_predict_step(self, lightning_model, batch_size, seq_len):
        """Check `predict_step` returns predictions without labels."""
        x_mlm = torch.randint(0, 125, (batch_size, seq_len))
        x_ssp = torch.randint(0, 125, (batch_size, seq_len))
        batch = (x_mlm, x_ssp)

        predictions = lightning_model.predict_step(batch, batch_idx=0)

        assert isinstance(predictions, tuple)
        assert len(predictions) == 2

        y_mlm_hat, y_ssp_hat = predictions

        # check mlm predictions
        assert isinstance(y_mlm_hat, torch.Tensor)
        assert y_mlm_hat.shape == (batch_size, seq_len, 125)

        # check ssp predictions
        assert isinstance(y_ssp_hat, torch.Tensor)
        assert y_ssp_hat.shape == (batch_size, seq_len, 8)

    @pytest.mark.parametrize("encoder_type", ["apta", "prot"])
    def test_configure_optimizers(self, mock_model, encoder_type):
        """Check that optimizer is configured correctly for both encoder types."""
        lightning_model = AptaTransEncoderLightning(
            mock_model, encoder_type=encoder_type
        )
        optimizer = lightning_model.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == lightning_model.lr
        assert optimizer.defaults["weight_decay"] == lightning_model.weight_decay
        assert optimizer.defaults["betas"] == lightning_model.betas
