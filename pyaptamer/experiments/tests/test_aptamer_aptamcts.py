"""Test suite for the AptamerEvalAptaMCTS experiment adapter."""

__author__ = ["agastya"]

import numpy as np
import pytest

from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.experiments._aptamer import BaseAptamerEval


class MockPipeline:
    """Mock pipeline with predict method for testing."""

    def __init__(self, fixed_score=0.9):
        self.fixed_score = fixed_score
        self.predict_calls = []

    def predict(self, aptamer: str, target: str) -> np.float64:
        self.predict_calls.append((aptamer, target))
        return np.float64(self.fixed_score)


class MockPipelineVariable:
    """Mock pipeline that returns scores based on input."""

    def predict(self, aptamer: str, target: str) -> np.float64:
        # Score based on GC content of aptamer
        gc_count = sum(1 for c in aptamer.upper() if c in "GC")
        total = max(len(aptamer), 1)
        return np.float64(gc_count / total)


class MockPipelineNoPredict:
    """Mock pipeline without predict method for error testing."""

    def evaluate(self, aptamer: str, target: str) -> np.float64:
        return np.float64(0.5)


# ---------------------------------------------------------------------------
# Inheritance and interface tests
# ---------------------------------------------------------------------------


class TestAptamerEvalAptaMCTSInterface:
    """Tests for AptamerEvalAptaMCTS interface compliance."""

    def test_inherits_from_base(self):
        """Check that AptamerEvalAptaMCTS inherits from BaseAptamerEval."""
        mock_pipeline = MockPipeline()
        experiment = AptamerEvalAptaMCTS(
            target="ACDEFGHIK", pipeline=mock_pipeline
        )

        assert isinstance(experiment, BaseAptamerEval)

    def test_has_required_methods(self):
        """Check that experiment has evaluate and _inputnames methods."""
        mock_pipeline = MockPipeline()
        experiment = AptamerEvalAptaMCTS(
            target="ACDEFGHIK", pipeline=mock_pipeline
        )

        assert hasattr(experiment, "evaluate")
        assert hasattr(experiment, "_inputnames")
        assert callable(experiment.evaluate)
        assert callable(experiment._inputnames)

    def test_inputnames(self):
        """Check that _inputnames returns correct input specification."""
        mock_pipeline = MockPipeline()
        experiment = AptamerEvalAptaMCTS(
            target="ACDEFGHIK", pipeline=mock_pipeline
        )

        names = experiment._inputnames()
        assert isinstance(names, list)
        assert "aptamer_candidate" in names


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestAptamerEvalAptaMCTSInit:
    """Tests for AptamerEvalAptaMCTS initialization."""

    def test_init_stores_target(self):
        """Check that target is stored correctly."""
        mock_pipeline = MockPipeline()
        experiment = AptamerEvalAptaMCTS(
            target="TESTTARGET", pipeline=mock_pipeline
        )

        assert experiment.target == "TESTTARGET"

    def test_init_stores_pipeline(self):
        """Check that pipeline is stored correctly."""
        mock_pipeline = MockPipeline()
        experiment = AptamerEvalAptaMCTS(
            target="ACDEF", pipeline=mock_pipeline
        )

        assert experiment.pipeline is mock_pipeline

    @pytest.mark.parametrize(
        "target",
        [
            "A",
            "ACDEF",
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTAYIAKQRQISFVKSHFSRQ",
        ],
    )
    def test_init_various_targets(self, target):
        """Check initialization with various target sequences."""
        mock_pipeline = MockPipeline()
        experiment = AptamerEvalAptaMCTS(target=target, pipeline=mock_pipeline)

        assert experiment.target == target


# ---------------------------------------------------------------------------
# evaluate() tests
# ---------------------------------------------------------------------------


class TestAptamerEvalAptaMCTSEvaluate:
    """Tests for AptamerEvalAptaMCTS.evaluate()."""

    def test_evaluate_returns_float64(self):
        """Check that evaluate returns np.float64."""
        mock_pipeline = MockPipeline(fixed_score=0.9)
        experiment = AptamerEvalAptaMCTS(
            target="ACDEFGHIK", pipeline=mock_pipeline
        )

        score = experiment.evaluate("ACGU")

        assert isinstance(score, np.float64)

    def test_evaluate_returns_correct_score(self):
        """Check that evaluate returns the score from pipeline.predict()."""
        mock_pipeline = MockPipeline(fixed_score=0.75)
        experiment = AptamerEvalAptaMCTS(
            target="ACDEFGHIK", pipeline=mock_pipeline
        )

        score = experiment.evaluate("ACGU")

        assert score == pytest.approx(0.75, abs=1e-6)

    def test_evaluate_calls_pipeline_predict(self):
        """Check that evaluate internally calls pipeline.predict()."""
        mock_pipeline = MockPipeline(fixed_score=0.8)
        experiment = AptamerEvalAptaMCTS(
            target="TESTTARGET", pipeline=mock_pipeline
        )

        _ = experiment.evaluate("ACGU")

        assert len(mock_pipeline.predict_calls) == 1
        aptamer, target = mock_pipeline.predict_calls[0]
        assert aptamer == "ACGU"
        assert target == "TESTTARGET"

    def test_evaluate_passes_correct_arguments(self):
        """Check that evaluate passes aptamer and target correctly."""
        mock_pipeline = MockPipeline()
        experiment = AptamerEvalAptaMCTS(
            target="PROTEIN", pipeline=mock_pipeline
        )

        _ = experiment.evaluate("RNA_SEQUENCE")

        aptamer, target = mock_pipeline.predict_calls[0]
        assert aptamer == "RNA_SEQUENCE"
        assert target == "PROTEIN"

    @pytest.mark.parametrize(
        "aptamer, target, expected_score",
        [
            ("A", "A", 0.0),  # 0 GC content
            ("GCGC", "ACDEF", 1.0),  # 100% GC content
            ("ACGU", "ACDEF", 0.5),  # 50% GC content
            ("GGGG", "M", 1.0),  # 100% GC content
        ],
    )
    def test_evaluate_variable_scores(self, aptamer, target, expected_score):
        """Check evaluate with pipeline that returns variable scores."""
        mock_pipeline = MockPipelineVariable()
        experiment = AptamerEvalAptaMCTS(target=target, pipeline=mock_pipeline)

        score = experiment.evaluate(aptamer)

        assert isinstance(score, np.float64)
        assert score == pytest.approx(expected_score, abs=1e-6)

    def test_evaluate_empty_aptamer(self):
        """Check evaluate with empty aptamer sequence."""
        mock_pipeline = MockPipelineVariable()
        experiment = AptamerEvalAptaMCTS(
            target="ACDEF", pipeline=mock_pipeline
        )

        score = experiment.evaluate("")

        assert isinstance(score, np.float64)
        # Empty sequence has 0 GC content
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_evaluate_short_aptamer(self):
        """Check evaluate with very short aptamer sequence."""
        mock_pipeline = MockPipeline(fixed_score=0.5)
        experiment = AptamerEvalAptaMCTS(
            target="A", pipeline=mock_pipeline
        )

        score = experiment.evaluate("A")

        assert isinstance(score, np.float64)
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_evaluate_long_aptamer(self):
        """Check evaluate with long aptamer sequence."""
        mock_pipeline = MockPipeline(fixed_score=0.85)
        experiment = AptamerEvalAptaMCTS(
            target="ACDEFGHIKLMNPQRSTVWY", pipeline=mock_pipeline
        )

        long_aptamer = "ACGU" * 25  # 100 nucleotides
        score = experiment.evaluate(long_aptamer)

        assert isinstance(score, np.float64)
        assert score == pytest.approx(0.85, abs=1e-6)

    def test_evaluate_deterministic(self):
        """Check that evaluate returns same score for same input."""
        mock_pipeline = MockPipeline(fixed_score=0.7)
        experiment = AptamerEvalAptaMCTS(
            target="ACDEF", pipeline=mock_pipeline
        )

        score1 = experiment.evaluate("ACGU")
        score2 = experiment.evaluate("ACGU")

        assert score1 == score2

    def test_evaluate_multiple_calls(self):
        """Check that evaluate can be called multiple times."""
        mock_pipeline = MockPipeline()
        experiment = AptamerEvalAptaMCTS(
            target="ACDEF", pipeline=mock_pipeline
        )

        scores = [experiment.evaluate(seq) for seq in ["AAAA", "GCGC", "ACGU"]]

        assert len(scores) == 3
        assert all(isinstance(s, np.float64) for s in scores)
        assert len(mock_pipeline.predict_calls) == 3

    @pytest.mark.parametrize(
        "target, aptamer",
        [
            ("ACDEF", "ACGU"),
            ("MKTAYIAKQRQISFVKSHFSRQ", "GCUAGCUA"),
            ("A", "U"),
        ],
    )
    def test_evaluate_various_combinations(self, target, aptamer):
        """Check evaluate works with various target-aptamer combinations."""
        mock_pipeline = MockPipeline(fixed_score=0.6)
        experiment = AptamerEvalAptaMCTS(target=target, pipeline=mock_pipeline)

        score = experiment.evaluate(aptamer)

        assert isinstance(score, np.float64)
        assert 0.0 <= score <= 1.0

    def test_evaluate_target_unchanged(self):
        """Check that evaluate does not modify the target attribute."""
        mock_pipeline = MockPipeline()
        original_target = "ACDEFGHIK"
        experiment = AptamerEvalAptaMCTS(
            target=original_target, pipeline=mock_pipeline
        )

        _ = experiment.evaluate("ACGU")

        assert experiment.target == original_target


# ---------------------------------------------------------------------------
# Integration tests with pipeline
# ---------------------------------------------------------------------------


class TestAptamerEvalAptaMCTSIntegration:
    """Integration tests for the experiment adapter with full pipeline."""

    def test_full_flow_evaluate_to_predict(self):
        """Check complete flow from experiment.evaluate to model.predict_proba."""
        from pyaptamer.aptamcts import AptaMCTSPipeline
        from pyaptamer.utils._aptamcts_utils import pairs_to_features

        class IntegrationMockModel:
            def __init__(self):
                self.call_count = 0

            def predict_proba(self, X):
                self.call_count += 1
                # Return probability based on first feature
                prob = float(X[0, 0]) if X.shape[0] > 0 else 0.5
                return np.array([[1 - prob, prob]], dtype=np.float64)

        mock_model = IntegrationMockModel()
        pipeline = AptaMCTSPipeline(model=mock_model)
        experiment = AptamerEvalAptaMCTS(
            target="ACDEF", pipeline=pipeline
        )

        score = experiment.evaluate("ACGU")

        assert isinstance(score, np.float64)
        assert mock_model.call_count == 1

    def test_experiment_isolation(self):
        """Check that multiple experiments with same pipeline are independent."""
        mock_pipeline = MockPipeline()
        exp1 = AptamerEvalAptaMCTS(target="TARGET1", pipeline=mock_pipeline)
        exp2 = AptamerEvalAptaMCTS(target="TARGET2", pipeline=mock_pipeline)

        _ = exp1.evaluate("ACGU")
        _ = exp2.evaluate("GCUA")

        assert len(mock_pipeline.predict_calls) == 2
        assert mock_pipeline.predict_calls[0] == ("ACGU", "TARGET1")
        assert mock_pipeline.predict_calls[1] == ("GCUA", "TARGET2")
