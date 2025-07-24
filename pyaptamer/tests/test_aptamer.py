"""Test suite for the experiment defined in the Aptamer experiment."""

__author__ = ["nennomp"]

import pytest
import torch
import torch.nn as nn

from pyaptamer.experiment._aptamer import Aptamer


class MockModel(nn.Module):
    def __init__(self, fixed_score=0.5):
        super().__init__()
        self.fixed_score = fixed_score

    def forward(self, x_apta, x_prot):
        # return a fixed score for deterministic testing
        return torch.tensor([self.fixed_score])

    def eval(self):
        pass


@pytest.fixture
def target_encoded():
    return torch.tensor([[1, 2, 3]])

@pytest.fixture
def target():
    return "ACGU"

@pytest.fixture
def model():
    return MockModel()

@pytest.fixture
def default_device():
    return torch.device("cpu")


class TestAptamer:
    """Test suite for the Aptamer() class."""

    @pytest.mark.parametrize(
        "device", 
        [
            (torch.device("cpu")),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA not available"
                )
            ),
        ]
    )
    def test_init(self, target_encoded, target, model, device):
        """Check correct initialization."""
        experiment = Aptamer(
            target_encoded=target_encoded,
            target=target,
            model=model,
            device=device,
        )
        assert experiment.target_encoded.device == device

    def test_inputnames(self, target_encoded, target, model, default_device):
        """Check that the inputs of the experiment are correctly returned."""
        experiment = Aptamer(
            target_encoded=target_encoded,
            target=target,
            model=model,
            device=default_device,
        )
        inputs = experiment._inputnames()
        assert inputs == ["aptamer_candidate"]

    @pytest.mark.parametrize(
        "device", 
        [
            (torch.device("cpu")),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA not available"
                )
            ),
        ]
    )
    def test_run(self, target_encoded, target, model, device):
        """Check that the experiment's evaluation method."""
        experiment = Aptamer(
            target_encoded=target_encoded,
            target=target,
            model=model,
            device=device,
        )
        
        aptamer_candidate = torch.tensor([[1, 2, 3]])
        score = experiment.run(aptamer_candidate)
        assert isinstance(score, torch.Tensor)
        assert score.shape == (1,)