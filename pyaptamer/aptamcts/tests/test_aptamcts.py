"""Test suite for the AptaMCTS pipeline and feature utilities."""

__author__ = ["agastya"]

import numpy as np
import pytest

from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.mcts import MCTS
from pyaptamer.utils._aptamcts_utils import pairs_to_features


class MockModel:
    """Mock model with predict_proba for testing."""

    def __init__(self, positive_prob=0.8):
        self.positive_prob = positive_prob

    def predict_proba(self, X):
        n_samples = X.shape[0]
        neg_prob = 1.0 - self.positive_prob
        return np.tile(
            np.array([[neg_prob, self.positive_prob]], dtype=np.float64),
            (n_samples, 1),
        )


class MockModelVariable:
    """Mock model that returns different probabilities based on input features."""

    def predict_proba(self, X):
        # Return probability proportional to first feature (deterministic for testing)
        probs = X[:, 0].clip(0, 1)
        neg_probs = 1.0 - probs
        return np.stack([neg_probs, probs], axis=1).astype(np.float64)


class MockModelNoProba:
    """Mock model without predict_proba for error testing."""

    def predict(self, X):
        return np.array([0.5])


class MockModelBadOutput:
    """Mock model that returns wrong shape from predict_proba."""

    def predict_proba(self, X):
        return np.array([0.5])


class MockModel1DOutput:
    """Mock model that returns 1D array instead of 2D."""

    def predict_proba(self, X):
        return np.array([0.5, 0.5])


# ---------------------------------------------------------------------------
# Feature encoding tests
# ---------------------------------------------------------------------------


class TestPairsToFeatures:
    """Tests for the pairs_to_features() utility."""

    def test_single_pair_shape_and_dtype(self):
        """Check that a single aptamer-target pair produces correct output shape."""
        X = pairs_to_features([("ACGT", "ACDEFGHIKLMNPQRSTVWY")])

        assert X.shape == (1, 27)
        assert X.dtype == np.float32

    @pytest.mark.parametrize(
        "pairs",
        [
            [("ACGT", "ACDEF"), ("GCUA", "KLMNP")],
            [("A", "A"), ("C", "C"), ("G", "G"), ("U", "U")],
        ],
    )
    def test_multiple_pairs_shape(self, pairs):
        """Check that multiple pairs produce correct batch shape."""
        X = pairs_to_features(pairs)

        assert X.shape == (len(pairs), 27)
        assert X.dtype == np.float32

    def test_empty_pairs(self):
        """Check that empty input produces correct empty array."""
        X = pairs_to_features([])

        assert X.shape == (0, 27)
        assert X.dtype == np.float32

    def test_dataframe_input(self):
        """Check that DataFrame input with aptamer/protein columns works."""
        import pandas as pd

        df = pd.DataFrame(
            {"aptamer": ["ACGT", "GCUA"], "protein": ["ACDEF", "KLMNP"]}
        )
        X = pairs_to_features(df)

        assert X.shape == (2, 27)
        assert X.dtype == np.float32

    def test_dna_to_rna_conversion(self):
        """Check that DNA sequences are converted to RNA before encoding."""
        # T should be converted to U, so features should match RNA version
        X_dna = pairs_to_features([("ACGT", "ACDEF")])
        X_rna = pairs_to_features([("ACGU", "ACDEF")])

        np.testing.assert_array_equal(X_dna, X_rna)

    def test_normalized_features_sum(self):
        """Check that character frequency features are normalized."""
        X = pairs_to_features([("AAAA", "ACDEF")])

        # First 5 columns are aptamer frequencies (A, C, G, U, N)
        # For "AAAA", A frequency should be 1.0
        assert X[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert X[0, 1:5].sum() == pytest.approx(0.0, abs=1e-5)

    def test_very_short_sequences(self):
        """Check encoding works for minimal sequences."""
        X = pairs_to_features([("A", "A")])

        assert X.shape == (1, 27)
        # Length features should be normalized
        assert X[0, -1] == pytest.approx(1.0, abs=1e-5)  # max(len, len) / max(len, len)

    def test_case_insensitivity(self):
        """Check that sequence encoding is case-insensitive."""
        X_upper = pairs_to_features([("ACGT", "ACDEF")])
        X_lower = pairs_to_features([("acgt", "acdef")])

        np.testing.assert_array_almost_equal(X_upper, X_lower)


# ---------------------------------------------------------------------------
# Pipeline predict() tests
# ---------------------------------------------------------------------------


class TestAptaMCTSPipelinePredict:
    """Tests for AptaMCTSPipeline.predict()."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with default mock model."""
        return AptaMCTSPipeline(model=MockModel(positive_prob=0.8))

    @pytest.fixture
    def pipeline_variable(self):
        """Create pipeline with variable-output mock model."""
        return AptaMCTSPipeline(model=MockModelVariable())

    def test_predict_returns_float64(self, pipeline):
        """Check that predict returns np.float64."""
        score = pipeline.predict(aptamer="ACGU", target="ACDEFGHIK")

        assert isinstance(score, np.float64)

    def test_predict_returns_correct_probability(self, pipeline):
        """Check that predict returns the positive-class probability."""
        score = pipeline.predict(aptamer="ACGU", target="ACDEFGHIK")

        assert score == pytest.approx(0.8, abs=1e-6)

    @pytest.mark.parametrize(
        "aptamer, target",
        [
            ("A", "A"),
            ("ACGU", "ACDEF"),
            ("ACGUACGUACGU", "ACDEFGHIKLMNPQRSTVWY"),
            ("U", "M"),
        ],
    )
    def test_predict_various_sequence_lengths(self, pipeline, aptamer, target):
        """Check predict works with various sequence lengths."""
        score = pipeline.predict(aptamer=aptamer, target=target)

        assert isinstance(score, np.float64)
        assert 0.0 <= score <= 1.0

    def test_predict_uses_predict_proba(self, pipeline_variable):
        """Check that predict internally uses model.predict_proba."""
        # Different sequences should produce different features and thus different scores
        score1 = pipeline_variable.predict(aptamer="AAAA", target="ACDEF")
        score2 = pipeline_variable.predict(aptamer="CCCC", target="ACDEF")

        # Features differ so scores should differ (unless by coincidence)
        assert isinstance(score1, np.float64)
        assert isinstance(score2, np.float64)

    def test_predict_no_predict_proba_raises(self):
        """Check that AttributeError is raised when model lacks predict_proba."""
        pipeline = AptaMCTSPipeline(model=MockModelNoProba())

        with pytest.raises(AttributeError, match="`model` must implement `predict_proba`"):
            pipeline.predict(aptamer="ACGU", target="ACDEF")

    def test_predict_bad_output_shape_raises(self):
        """Check that ValueError is raised when predict_proba returns wrong shape."""
        pipeline = AptaMCTSPipeline(model=MockModelBadOutput())

        with pytest.raises(ValueError, match="`predict_proba` must return an array"):
            pipeline.predict(aptamer="ACGU", target="ACDEF")

    def test_predict_1d_output_raises(self):
        """Check that ValueError is raised when predict_proba returns 1D array."""
        pipeline = AptaMCTSPipeline(model=MockModel1DOutput())

        with pytest.raises(ValueError, match="`predict_proba` must return an array"):
            pipeline.predict(aptamer="ACGU", target="ACDEF")

    def test_predict_deterministic(self, pipeline):
        """Check that predict returns same score for same input."""
        score1 = pipeline.predict(aptamer="ACGU", target="ACDEFGHIK")
        score2 = pipeline.predict(aptamer="ACGU", target="ACDEFGHIK")

        assert score1 == score2

    def test_predict_different_aptamers(self, pipeline_variable):
        """Check that different aptamers produce different scores."""
        score1 = pipeline_variable.predict(aptamer="AAAA", target="ACDEF")
        score2 = pipeline_variable.predict(aptamer="GCGC", target="ACDEF")

        # Aptamer features differ (GC content), so scores should differ
        assert score1 != score2


# ---------------------------------------------------------------------------
# Pipeline recommend() tests
# ---------------------------------------------------------------------------


class TestAptaMCTSPipelineRecommend:
    """Tests for AptaMCTSPipeline.recommend()."""

    def test_recommend_returns_set(self, monkeypatch):
        """Check that recommend returns a set of candidates."""
        pipeline = AptaMCTSPipeline(model=MockModel(), depth=5, n_iterations=2)

        class MockMCTS:
            def __init__(self, **kwargs):
                self.counter = 0

            def run(self, verbose=False):
                candidate = f"APT{self.counter:03d}"
                result = {
                    "candidate": candidate,
                    "sequence": f"{candidate}_SEQ",
                    "score": np.float64(0.5),
                }
                self.counter += 1
                return result

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        candidates = pipeline.recommend(target="ACDEFGHIK", n_candidates=3)

        assert isinstance(candidates, set)

    def test_recommend_correct_number_of_candidates(self, monkeypatch):
        """Check that recommend returns exactly n_candidates unique results."""
        pipeline = AptaMCTSPipeline(model=MockModel(), depth=5, n_iterations=2)

        class MockMCTS:
            def __init__(self, **kwargs):
                self.counter = 0

            def run(self, verbose=False):
                candidate = f"APT{self.counter:03d}"
                result = {
                    "candidate": candidate,
                    "sequence": f"{candidate}_SEQ",
                    "score": np.float64(0.5),
                }
                self.counter += 1
                return result

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        for n in [1, 3, 5, 10]:
            candidates = pipeline.recommend(target="ACDEFGHIK", n_candidates=n)
            assert len(candidates) == n

    def test_recommend_tuple_structure(self, monkeypatch):
        """Check that each candidate is a (candidate, sequence, score) tuple."""
        pipeline = AptaMCTSPipeline(model=MockModel(), depth=5, n_iterations=2)

        class MockMCTS:
            def __init__(self, **kwargs):
                self.counter = 0

            def run(self, verbose=False):
                candidate = f"APT{self.counter:03d}"
                result = {
                    "candidate": candidate,
                    "sequence": f"{candidate}_SEQ",
                    "score": np.float64(0.5 + self.counter * 0.1),
                }
                self.counter += 1
                return result

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        candidates = pipeline.recommend(target="ACDEFGHIK", n_candidates=3)

        for candidate, sequence, score in candidates:
            assert isinstance(candidate, str)
            assert isinstance(sequence, str)
            assert isinstance(score, float)

    def test_recommend_uses_mcts(self, monkeypatch):
        """Check that recommend actually calls MCTS with correct parameters."""
        mcts_calls = []

        class MockMCTS:
            def __init__(self, experiment, depth, n_iterations):
                mcts_calls.append(
                    {"depth": depth, "n_iterations": n_iterations}
                )
                self.counter = 0

            def run(self, verbose=False):
                result = {
                    "candidate": "APT000",
                    "sequence": "APT000_SEQ",
                    "score": np.float64(0.5),
                }
                self.counter += 1
                return result

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        pipeline = AptaMCTSPipeline(model=MockModel(), depth=10, n_iterations=500)
        pipeline.recommend(target="ACDEF", n_candidates=1)

        assert len(mcts_calls) >= 1
        assert mcts_calls[0]["depth"] == 10
        assert mcts_calls[0]["n_iterations"] == 500

    def test_recommend_uses_experiment_adapter(self, monkeypatch):
        """Check that recommend creates AptamerEvalAptaMCTS experiment."""
        experiment_created = []

        original_init = AptamerEvalAptaMCTS.__init__

        def mock_init(self, target, pipeline):
            experiment_created.append({"target": target, "pipeline": pipeline})
            original_init(self, target, pipeline)

        monkeypatch.setattr(
            AptamerEvalAptaMCTS, "__init__", mock_init
        )

        class MockMCTS:
            def __init__(self, **kwargs):
                self.counter = 0

            def run(self, verbose=False):
                return {
                    "candidate": "APT000",
                    "sequence": "APT000_SEQ",
                    "score": np.float64(0.5),
                }

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        pipeline = AptaMCTSPipeline(model=MockModel(), depth=5, n_iterations=2)
        pipeline.recommend(target="TESTTARGET", n_candidates=1)

        assert len(experiment_created) == 1
        assert experiment_created[0]["target"] == "TESTTARGET"
        assert experiment_created[0]["pipeline"] is pipeline

    def test_recommend_deduplicates(self, monkeypatch):
        """Check that recommend handles duplicate candidates from MCTS."""
        call_count = 0

        class MockMCTS:
            def __init__(self, **kwargs):
                pass

            def run(self, verbose=False):
                nonlocal call_count
                call_count += 1
                # Return same candidate first 3 times, then different ones
                if call_count <= 3:
                    return {
                        "candidate": "APT000",
                        "sequence": "APT000_SEQ",
                        "score": np.float64(0.5),
                    }
                else:
                    idx = call_count - 3
                    return {
                        "candidate": f"APT{idx:03d}",
                        "sequence": f"APT{idx:03d}_SEQ",
                        "score": np.float64(0.5),
                    }

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        pipeline = AptaMCTSPipeline(model=MockModel(), depth=5, n_iterations=2)
        candidates = pipeline.recommend(target="ACDEF", n_candidates=2)

        assert len(candidates) == 2
        # Should have called MCTS more than 2 times due to deduplication
        assert call_count > 2

    def test_recommend_score_converted_to_float(self, monkeypatch):
        """Check that numpy scores are converted to Python floats."""

        class MockMCTS:
            def __init__(self, **kwargs):
                pass

            def run(self, verbose=False):
                return {
                    "candidate": "APT000",
                    "sequence": "APT000_SEQ",
                    "score": np.float32(0.75),
                }

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        pipeline = AptaMCTSPipeline(model=MockModel(), depth=5, n_iterations=2)
        candidates = pipeline.recommend(target="ACDEF", n_candidates=1)

        for _, _, score in candidates:
            assert type(score) is float

    @pytest.mark.parametrize("n_candidates", [1, 5, 10])
    def test_recommend_consistency_with_n_candidates(self, monkeypatch, n_candidates):
        """Check recommend returns exactly the requested number of candidates."""
        counter = 0

        class MockMCTS:
            def __init__(self, **kwargs):
                pass

            def run(self, verbose=False):
                nonlocal counter
                counter += 1
                return {
                    "candidate": f"CAND{counter:03d}",
                    "sequence": f"SEQ{counter:03d}",
                    "score": np.float64(counter * 0.1),
                }

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        pipeline = AptaMCTSPipeline(model=MockModel(), depth=5, n_iterations=2)
        candidates = pipeline.recommend(target="ACDEF", n_candidates=n_candidates)

        assert len(candidates) == n_candidates
        counter = 0  # reset for next parametrization


# ---------------------------------------------------------------------------
# Pipeline initialization tests
# ---------------------------------------------------------------------------


class TestAptaMCTSPipelineInit:
    """Tests for AptaMCTSPipeline initialization."""

    def test_default_parameters(self):
        """Check default depth and n_iterations values."""
        pipeline = AptaMCTSPipeline(model=MockModel())

        assert pipeline.depth == 20
        assert pipeline.n_iterations == 1000

    @pytest.mark.parametrize("depth, n_iterations", [(5, 100), (10, 500), (50, 2000)])
    def test_custom_parameters(self, depth, n_iterations):
        """Check custom depth and n_iterations are stored correctly."""
        pipeline = AptaMCTSPipeline(
            model=MockModel(), depth=depth, n_iterations=n_iterations
        )

        assert pipeline.depth == depth
        assert pipeline.n_iterations == n_iterations

    def test_model_stored(self):
        """Check that model is stored as attribute."""
        model = MockModel()
        pipeline = AptaMCTSPipeline(model=model)

        assert pipeline.model is model

    def test_init_aptamer_experiment_type(self):
        """Check _init_aptamer_experiment returns AptamerEvalAptaMCTS."""
        pipeline = AptaMCTSPipeline(model=MockModel())
        experiment = pipeline._init_aptamer_experiment(target="ACDEF")

        assert isinstance(experiment, AptamerEvalAptaMCTS)
        assert experiment.target == "ACDEF"
        assert experiment.pipeline is pipeline
