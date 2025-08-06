"""Test suite for the Aptamer experiment class."""

__author__ = ["nennomp"]

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pyaptamer.experiments import Aptamer


class MockModel(nn.Module):
    def __init__(self, fixed_score=0.5):
        super().__init__()
        self.fixed_score = fixed_score

    def forward_imap(self, x_apta, x_prot):
        # return a randomly generated interaction map for testing
        return torch.randn(1, 1, x_apta.size(1), x_prot.size(1))

    def forward(self, x_apta, x_prot):
        # return a fixed score for deterministic testing
        return torch.tensor([self.fixed_score])

    def eval(self):
        pass


@pytest.fixture
def experiment():
    target_encoded = torch.tensor([[1, 2, 3, 0, 0, 0]])
    target = "DHRNE"
    model = MockModel()
    device = torch.device("cpu")
    return Aptamer(
        target_encoded=target_encoded,
        target=target,
        model=model,
        device=device,
    )


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
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_init(self, target_encoded, target, model, device):
        """Check correct initialization."""
        experiment = Aptamer(
            target_encoded=target_encoded,
            target=target,
            model=model,
            device=device,
        )
        assert experiment.target_encoded.device.type == device.type

    def test_inputnames(self, experiment):
        """Check that the inputs of the experiment are correctly returned."""
        inputs = experiment._inputnames()
        assert inputs == ["aptamer_candidate"]

    def test_reconstruct(self, experiment):
        """Check sequence reconstruction."""
        # empty sequence
        assert torch.equal(experiment._reconstruct(""), torch.tensor([]))
        # prepend and append
        result = experiment._reconstruct("A_C__G_U")
        assert result.shape == (1, 275)
        assert result[0, 0] != 0  # first triplet should not be 0
        assert result[0, 1] != 0  # second triplet should not be 0
        assert torch.all(result[0, 2:] == 0)  # rest should be padding
        # already reconstructed sequence
        result = experiment._reconstruct("ACGU")
        assert result.shape == (1, 275)
        assert result[0, 0] != 0  # first triplet should not be 0
        assert result[0, 1] != 0  # second triplet should not be 0
        assert torch.all(result[0, 2:] == 0)  # rest should be padding

    def test_evaluate(self, experiment):
        """Check that the experiment's evaluation method works correctly."""
        aptamer_candidate = "A_C_GU"
        score = experiment.evaluate(aptamer_candidate)
        assert isinstance(score, Tensor)
        assert score.shape == (1,)

    def test_evaluate_imap(self, experiment):
        """
        Check that the experiment's evaluation method works correctly when returning
        interaction map.
        """
        aptamer_candidate = "ACGU"
        score = experiment.evaluate(aptamer_candidate, return_interaction_map=True)
        assert isinstance(score, Tensor)
        # 275 is the default maximum length in the rna2vec encoding, and thus the
        # encoded aptamer sequence length
        assert score.shape == (1, 1, 275, experiment.target_encoded.shape[1])
