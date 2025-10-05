"""Test suite for the AptamerEval experiment classes."""

__author__ = ["nennomp"]

import numpy
import pytest
import torch
import torch.nn as nn

from pyaptamer.experiments import AptamerEvalAptaNet, AptamerEvalAptaTrans


class MockModel(nn.Module):
    def __init__(self, fixed_score=0.5):
        super().__init__()
        self.fixed_score = fixed_score
        # mock embeddings with required attributes
        self.apta_embedding = type("MockEmbedding", (), {"max_len": 100})()
        self.prot_embedding = type("MockEmbedding", (), {"max_len": 150})()

    def forward_imap(self, x_apta, x_prot):
        # return a randomly generated interaction map for testing
        return torch.randn(1, 1, x_apta.size(1), x_prot.size(1))

    def forward(self, x_apta, x_prot):
        # return a fixed score for deterministic testing
        return torch.tensor([self.fixed_score])

    def eval(self):
        pass


class MockAptaNetPipeline:
    """Mock AptaNetPipeline for testing."""

    def __init__(self, fixed_score=0.7):
        self.fixed_score = fixed_score
        self.is_fitted = True

    def predict_proba(self, X):
        """
        Mock predict method that returns fixed scores for binary classification (no
        binding, binding).
        """
        # return probability scores as a list
        return numpy.array([[1 - self.fixed_score, self.fixed_score]] * len(X))

    def fit(self, X, y):
        """Mock fit method."""
        self.is_fitted = True
        return self


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def pipeline():
    return MockAptaNetPipeline()


@pytest.fixture
def device():
    """Default device for AptaTrans experiments."""
    return torch.device("cpu")


@pytest.fixture
def target():
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def prot_words():
    """Protein words for AptaTrans experiments."""
    return {"AAA": 0.5, "AAC": 0.3, "AAG": 0.2, "AUG": 0.1, "CGA": 0.4}


class TestAptamerEvalConcrete:
    """Test suite for concrete AptamerEval implementations (AptaTrans and AptaNet)."""

    @pytest.fixture
    def experiment(self, request, target, model, device, prot_words, pipeline):
        """
        Fixture that returns an initialized AptamerEval instance based on the parameter.
        """
        if request.param == "aptatrans":
            return AptamerEvalAptaTrans(
                target=target,
                model=model,
                device=device,
                prot_words=prot_words,
            )
        elif request.param == "aptanet":
            return AptamerEvalAptaNet(
                target=target,
                pipeline=pipeline,
            )

    @pytest.fixture
    def aptatrans_experiment(self, target, model, device, prot_words):
        """Fixture that returns an initialized AptamerEvalAptaTrans instance."""
        return AptamerEvalAptaTrans(
            target=target,
            model=model,
            device=device,
            prot_words=prot_words,
        )

    @pytest.fixture
    def aptanet_experiment(self, target, pipeline):
        """Fixture that returns an initialized AptamerEvalAptaNet instance."""
        return AptamerEvalAptaNet(
            target=target,
            pipeline=pipeline,
        )

    @pytest.mark.parametrize("experiment", ["aptatrans", "aptanet"], indirect=True)
    def test_evaluate(self, experiment):
        """Check that the experiment's evaluation method works correctly."""
        aptamer_candidate = "ACGU"
        score = experiment.evaluate(aptamer_candidate)
        assert isinstance(score, numpy.float64)

    @pytest.mark.parametrize("experiment", ["aptatrans", "aptanet"], indirect=True)
    def test_evaluate_empty_sequence(self, experiment):
        """Check evaluation with empty sequence."""
        score = experiment.evaluate("")
        assert isinstance(score, numpy.float64)

    @pytest.mark.parametrize("experiment", ["aptatrans", "aptanet"], indirect=True)
    def test_evaluate_with_underscores(self, experiment):
        """Check evaluation with sequences containing underscores."""
        aptamer_candidate = "ACGU"
        score = experiment.evaluate(aptamer_candidate)
        assert isinstance(score, numpy.float64)

    def test_evaluate_imap(self, aptatrans_experiment):
        """
        Check that the experiment's evaluation method works correctly when returning
        interaction map.
        """
        aptamer_candidate = "ACGU"
        score = aptatrans_experiment.evaluate(
            aptamer_candidate, return_interaction_map=True
        )
        assert isinstance(score, numpy.ndarray)
        # 100 is the maximum length specified in our mock model
        assert score.shape == (1, 1, 100, aptatrans_experiment.target_encoded.shape[1])
