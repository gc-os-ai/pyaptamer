"""Test suite for the AptamerEval experiment classes."""

__author__ = ["nennomp", "siddharth7113"]

import numpy
import pytest

from pyaptamer.experiments import AptamerEvalAptaNet, AptamerEvalAptaTrans


class MockPipeline:
    """Mock fitted pipeline with fixed binding probabilities."""

    def __init__(self, fixed_score=0.7, aptamer_col="aptamer", protein_col="protein"):
        self.fixed_score = fixed_score
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        self.is_fitted = True

    def predict_proba(self, X):
        """
        Mock predict method that returns fixed scores for binary classification (no
        binding, binding).

        ``X`` is a :class:`~pyaptamer.data.loader.MoleculeLoader`; materialize it
        to learn how many pairs to score.
        """
        n = len(X.to_dataframe())
        # return probability scores as a list
        return numpy.array([[1 - self.fixed_score, self.fixed_score]] * n)

    def fit(self, X, y):
        """Mock fit method."""
        self.is_fitted = True
        return self


@pytest.fixture
def pipeline():
    return MockPipeline()


@pytest.fixture
def target():
    return "ACDEFGHIKLMNPQRSTVWY"


class TestAptamerEvalConcrete:
    """Test suite for concrete AptamerEval implementations (AptaTrans and AptaNet)."""

    @pytest.fixture
    def experiment(self, request, target, pipeline):
        """Return an AptamerEval instance of the class named by the parameter."""
        return request.param(target=target, pipeline=pipeline)

    @pytest.mark.parametrize(
        "experiment", [AptamerEvalAptaNet, AptamerEvalAptaTrans], indirect=True
    )
    def test_evaluate(self, experiment):
        """Check that the experiment's evaluation method works correctly."""
        aptamer_candidate = "ACGU"
        score = experiment.evaluate(aptamer_candidate)
        assert isinstance(score, numpy.float64)


class RecordingPipeline(MockPipeline):
    """Mock pipeline that records the column names it is asked to score."""

    def predict_proba(self, X):
        self.columns = X.to_dataframe().columns.tolist()
        return super().predict_proba(X)


@pytest.mark.parametrize("eval_cls", [AptamerEvalAptaNet, AptamerEvalAptaTrans])
def test_evaluate_uses_the_pipeline_column_names(eval_cls, target):
    """The pair is built with the column names the pipeline was configured with."""
    pipeline = RecordingPipeline(aptamer_col="apt", protein_col="target")
    eval_cls(target=target, pipeline=pipeline).evaluate("ACGU")
    assert pipeline.columns == ["apt", "target"]
