"""Test suite for the AptamerEval experiment classes."""

__author__ = ["nennomp"]

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pyaptamer.experiments._aptamer_eval import BaseAptamerEval
from pyaptamer.experiments import (
    AptamerEvalAptaTrans, 
    AptamerEvalAptaNet
)


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
    
    def predict(self, X, output_type="proba"):
        """Mock predict method that returns fixed scores."""
        if output_type == "proba":
            # return probability scores as a list
            return [self.fixed_score] * len(X)
        else:
            # return binary predictions
            return [1 if self.fixed_score > 0.5 else 0] * len(X)
    
    def fit(self, X, y):
        """Mock fit method."""
        self.is_fitted = True
        return self


@pytest.fixture
def target():
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def target_encoded():
    return torch.tensor([[1, 2, 3]])


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def default_device():
    return torch.device("cpu")


@pytest.fixture
def prot_words():
    return {"AAA": 0.5, "AAC": 0.3, "AAG": 0.2, "AUG": 0.1, "CGA": 0.4}


@pytest.fixture
def pipeline():
    """Create a mock pipeline for testing."""
    return MockAptaNetPipeline()


@pytest.fixture
def aptatrans_experiment(target, model, default_device, prot_words):
    return AptamerEvalAptaTrans(
        target=target,
        model=model,
        device=default_device,
        prot_words=prot_words,
    )


@pytest.fixture
def aptanet_experiment(target, pipeline):
    """Create an AptamerEvalAptaNet instance for testing."""
    return AptamerEvalAptaNet(target=target, pipeline=pipeline)


class TestBaseAptamerEval:
    """Test suite for the BaseAptamerEval abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Check that the abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAptamerEval("ACGT")
    
    def test_inputnames(self, aptanet_experiment):
        """Check that the inputs of the experiment are correctly returned."""
        inputs = aptanet_experiment._inputnames()
        assert inputs == ["aptamer_candidate"]
    
    @pytest.mark.parametrize(
        "sequence, expected",
        [
            ("", ""), # empty sequence
            ("ACGU", "ACGU"), # already reconstructed
            ("A_C_G_U_", "UGCA"), # all prepended
            ("_A_C_G_U", "ACGU"), # all appended
            ("A__C", "AC"), # mixed prepend and append
            ("A_U_G_G_C_", "CGGUA"), # complex mixed sequence
        ],
    )
    def test_base_reconstruct(self, aptanet_experiment, sequence, expected):
        """Check sequence reconstruction with various inputs."""
        result = aptanet_experiment.reconstruct(sequence)
        assert result == expected
    
    def test_base_reconstruct_odd_length_error(self, aptanet_experiment):
        """Check that odd length encoded sequences raise an assertion error."""
        with pytest.raises(AssertionError, match="Encoded sequence must have even length"):
            aptanet_experiment.reconstruct("A_C") # length 3, should be even


class TestAptamerEvalAptaTrans:
    """Test suite for the AptamerEvalAptaTrans class."""

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
    def test_init(self, target, model, device, prot_words):
        """Check correct initialization."""
        experiment = AptamerEvalAptaTrans(
            target=target,
            model=model,
            device=device,
            prot_words=prot_words,
        )
        assert experiment.target_encoded.device.type == device.type

    def test_inputnames(self, aptatrans_experiment):
        """Check that the inputs of the experiment are correctly returned."""
        inputs = aptatrans_experiment._inputnames()
        assert inputs == ["aptamer_candidate"]

    def test_reconstruct(self, aptatrans_experiment):
        """Check sequence reconstruction."""
        # empty sequence
        result_str, result_vector = aptatrans_experiment.reconstruct("")
        assert result_str == ""
        assert torch.equal(result_vector, torch.tensor([]))
        # prepend and append
        result_str, result_vector = aptatrans_experiment.reconstruct("A_C__G_U")
        assert result_str == "CAGU"
        # 100 is the maximum length specified in our mock model
        assert result_vector.shape == (1, 100)
        assert result_vector[0, 0] != 0  # first triplet should not be 0
        assert result_vector[0, 1] != 0  # second triplet should not be 0
        assert torch.all(result_vector[0, 2:] == 0)  # rest should be padding

    def test_evaluate(self, aptatrans_experiment):
        """Check that the experiment's evaluation method works correctly."""
        aptamer_candidate = "A_C_GU"
        score = aptatrans_experiment.evaluate(aptamer_candidate)
        assert isinstance(score, Tensor)
        assert score.shape == (1,)

    def test_evaluate_imap(self, aptatrans_experiment):
        """
        Check that the experiment's evaluation method works correctly when returning
        interaction map.
        """
        aptamer_candidate = "ACGU"
        score = aptatrans_experiment.evaluate(aptamer_candidate, return_interaction_map=True)
        assert isinstance(score, Tensor)
        # 100 is the maximum length specified in our mock model
        assert score.shape == (1, 1, 100, aptatrans_experiment.target_encoded.shape[1])


class TestAptamerEvalAptaNet:
    """Test suite for the AptamerEvalAptaNet class."""
    
    def test_init(self, target, pipeline):
        """Check correct initialization."""
        experiment = AptamerEvalAptaNet(target=target, pipeline=pipeline)
        assert experiment.target == target
        assert experiment.pipeline == pipeline
    
    def test_inputnames(self, aptanet_experiment):
        """Check that the inputs of the experiment are correctly returned."""
        inputs = aptanet_experiment._inputnames()
        assert inputs == ["aptamer_candidate"]
    
    @pytest.mark.parametrize(
        "sequence, expected",
        [
            ("", ""), # empty sequence
            ("ACGU", "ACGU"), # already reconstructed
            ("A_C_G_U_", "UGCA"), # all prepended
            ("_A_C_G_U", "ACGU"), # all appended
            ("A__C", "AC"), # mixed prepend and append
            ("A_U_G_G_C_", "CGGUA"), # complex mixed sequence
        ],
    )
    def test_reconstruct(self, aptanet_experiment, sequence, expected):
        """Check sequence reconstruction with various inputs."""
        result = aptanet_experiment.reconstruct(sequence)
        assert result == expected
    
    def test_reconstruct_odd_length_error(self, aptanet_experiment):
        """Check that odd length encoded sequences raise an assertion error."""
        with pytest.raises(AssertionError, match="Encoded sequence must have even length"):
            aptanet_experiment.reconstruct("A_C") # length 3, should be even
    
    def test_evaluate(self, aptanet_experiment):
        """Check that the experiment's evaluation method works correctly."""
        aptamer_candidate = "A_C_G_U_"
        score = aptanet_experiment.evaluate(aptamer_candidate)
        assert isinstance(score, Tensor)
        assert score.shape == (1,)
        assert 0 <= score.item() <= 1 # probability should be between 0 and 1
    
    def test_evaluate_empty_sequence(self, aptanet_experiment):
        """Check evaluation with empty sequence."""
        score = aptanet_experiment.evaluate("")
        assert isinstance(score, Tensor)
        assert score.shape == (1,)
    
    def test_evaluate_already_reconstructed(self, aptanet_experiment):
        """Check evaluation with already reconstructed sequence."""
        aptamer_candidate = "ACGU" # no underscores
        score = aptanet_experiment.evaluate(aptamer_candidate)
        assert isinstance(score, Tensor)
        assert score.shape == (1,)
    
    def test_evaluate_different_scores(self, target):
        """Check that different pipelines return different scores."""
        pipeline1 = MockAptaNetPipeline(fixed_score=0.3)
        pipeline2 = MockAptaNetPipeline(fixed_score=0.8)
        
        experiment1 = AptamerEvalAptaNet(target=target, pipeline=pipeline1)
        experiment2 = AptamerEvalAptaNet(target=target, pipeline=pipeline2)
        
        aptamer_candidate = "A_C_G_U_"
        score1 = experiment1.evaluate(aptamer_candidate)
        score2 = experiment2.evaluate(aptamer_candidate)
        
        assert score1.item() != score2.item()
        assert round(score1.item(), 1) == 0.3
        assert round(score2.item(), 1) == 0.8
    
    def test_pipeline_called_correctly(self, target):
        """Check that the pipeline is called with correct arguments."""
        class SpyPipeline(MockAptaNetPipeline):
            def __init__(self):
                super().__init__()
                self.last_call_args = None
                self.last_call_kwargs = None
            
            def predict(self, X, output_type="proba"):
                self.last_call_args = (X,)
                self.last_call_kwargs = {"output_type": output_type}
                return super().predict(X, output_type)
        
        spy_pipeline = SpyPipeline()
        experiment = AptamerEvalAptaNet(target=target, pipeline=spy_pipeline)
        
        aptamer_candidate = "A_U_G_C_"
        experiment.evaluate(aptamer_candidate)
        
        # check that predict was called with correct arguments
        assert spy_pipeline.last_call_args is not None
        assert len(spy_pipeline.last_call_args[0]) == 1  # one sequence pair
        aptamer_seq, target_seq = spy_pipeline.last_call_args[0][0]
        assert aptamer_seq == "CGUA"  # reconstructed sequence
        assert target_seq == target
        assert spy_pipeline.last_call_kwargs["output_type"] == "proba"
    
    @pytest.mark.parametrize(
        "aptamer_candidate",
        [
            "A_C_G_U_",
            "_A_C_G_U",
            "A__C_G_U",
            "",
            "ACGU",
        ],
    )
    def test_evaluate_various_candidates(self, aptanet_experiment, aptamer_candidate):
        """Test evaluation with various aptamer candidate formats."""
        score = aptanet_experiment.evaluate(aptamer_candidate)
        assert isinstance(score, Tensor)
        assert score.shape == (1,)