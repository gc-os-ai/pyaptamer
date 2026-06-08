"""Test suite for the AptaMCTS pipeline and feature utilities."""

__author__ = ["agastya"]

import numpy as np
import pytest

from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.mcts import MCTS
from pyaptamer.utils._aptamcts_utils import pairs_to_features


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
        X_dna = pairs_to_features([("ACGT", "ACDEF")])
        X_rna = pairs_to_features([("ACGU", "ACDEF")])

        np.testing.assert_array_equal(X_dna, X_rna)

    def test_normalized_features_sum(self):
        """Check that character frequency features are normalized."""
        X = pairs_to_features([("AAAA", "ACDEF")])

        assert X[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert X[0, 1:5].sum() == pytest.approx(0.0, abs=1e-5)

    def test_very_short_sequences(self):
        """Check encoding works for minimal sequences."""
        X = pairs_to_features([("A", "A")])

        assert X.shape == (1, 27)
        assert X[0, -1] == pytest.approx(1.0, abs=1e-5)

    def test_case_insensitivity(self):
        """Check that sequence encoding is case-insensitive."""
        X_upper = pairs_to_features([("ACGT", "ACDEF")])
        X_lower = pairs_to_features([("acgt", "acdef")])

        np.testing.assert_array_almost_equal(X_upper, X_lower)


# ---------------------------------------------------------------------------
# Pipeline fit() tests
# ---------------------------------------------------------------------------


class TestAptaMCTSPipelineFit:
    """Tests for AptaMCTSPipeline.fit()."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
        protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
        X = [(aptamer, protein) for _ in range(40)]
        y = np.array([0] * 20 + [1] * 20, dtype=np.float32)
        return X, y

    def test_fit_returns_self(self, sample_data):
        """Check that fit returns self for method chaining."""
        pipeline = AptaMCTSPipeline()
        X, y = sample_data

        result = pipeline.fit(X, y)

        assert result is pipeline

    def test_fit_creates_pipeline_attribute(self, sample_data):
        """Check that fit creates the pipeline_ attribute."""
        pipeline = AptaMCTSPipeline()
        X, y = sample_data

        pipeline.fit(X, y)

        assert hasattr(pipeline, "pipeline_")

    def test_fit_without_estimator_uses_default(self, sample_data):
        """Check that fit works with default estimator."""
        pipeline = AptaMCTSPipeline()
        X, y = sample_data

        pipeline.fit(X, y)

        assert pipeline.pipeline_ is not None

    def test_fit_with_custom_estimator(self, sample_data):
        """Check that fit works with custom estimator."""
        from sklearn.ensemble import GradientBoostingClassifier

        pipeline = AptaMCTSPipeline(estimator=GradientBoostingClassifier(n_estimators=10))
        X, y = sample_data

        pipeline.fit(X, y)

        assert pipeline.pipeline_ is not None

    def test_fit_stores_parameters(self):
        """Check that depth and n_iterations are stored."""
        pipeline = AptaMCTSPipeline(depth=10, n_iterations=500)

        assert pipeline.depth == 10
        assert pipeline.n_iterations == 500


# ---------------------------------------------------------------------------
# Pipeline predict_proba() tests
# ---------------------------------------------------------------------------


class TestAptaMCTSPipelinePredictProba:
    """Tests for AptaMCTSPipeline.predict_proba()."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline for testing."""
        aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
        protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
        X = [(aptamer, protein) for _ in range(40)]
        y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

        pipeline = AptaMCTSPipeline()
        pipeline.fit(X, y)
        return pipeline, aptamer, protein

    def test_predict_proba_returns_ndarray(self, fitted_pipeline):
        """Check that predict_proba returns numpy array."""
        pipeline, aptamer, protein = fitted_pipeline
        X = [(aptamer, protein) for _ in range(10)]

        proba = pipeline.predict_proba(X)

        assert isinstance(proba, np.ndarray)

    def test_predict_proba_shape(self, fitted_pipeline):
        """Check that predict_proba returns correct shape."""
        pipeline, aptamer, protein = fitted_pipeline
        X = [(aptamer, protein) for _ in range(10)]

        proba = pipeline.predict_proba(X)

        assert proba.shape == (10, 2)

    def test_predict_proba_values_in_range(self, fitted_pipeline):
        """Check that probabilities are in [0, 1] range."""
        pipeline, aptamer, protein = fitted_pipeline
        X = [(aptamer, protein) for _ in range(10)]

        proba = pipeline.predict_proba(X)

        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_predict_proba_rows_sum_to_one(self, fitted_pipeline):
        """Check that probability rows sum to 1."""
        pipeline, aptamer, protein = fitted_pipeline
        X = [(aptamer, protein) for _ in range(10)]

        proba = pipeline.predict_proba(X)

        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10))

    def test_predict_proba_unfitted_raises(self):
        """Check that predict_proba raises when not fitted."""
        pipeline = AptaMCTSPipeline()

        with pytest.raises(Exception):
            pipeline.predict_proba([("ACGU", "ACDEF")])


# ---------------------------------------------------------------------------
# Pipeline predict() tests
# ---------------------------------------------------------------------------


class TestAptaMCTSPipelinePredict:
    """Tests for AptaMCTSPipeline.predict()."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline for testing."""
        aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
        protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
        X = [(aptamer, protein) for _ in range(40)]
        y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

        pipeline = AptaMCTSPipeline()
        pipeline.fit(X, y)
        return pipeline, aptamer, protein

    def test_predict_returns_ndarray(self, fitted_pipeline):
        """Check that predict returns numpy array."""
        pipeline, aptamer, protein = fitted_pipeline
        X = [(aptamer, protein) for _ in range(10)]

        preds = pipeline.predict(X)

        assert isinstance(preds, np.ndarray)

    def test_predict_shape(self, fitted_pipeline):
        """Check that predict returns correct shape."""
        pipeline, aptamer, protein = fitted_pipeline
        X = [(aptamer, protein) for _ in range(10)]

        preds = pipeline.predict(X)

        assert preds.shape == (10,)

    def test_predict_returns_valid_labels(self, fitted_pipeline):
        """Check that predict returns valid class labels."""
        pipeline, aptamer, protein = fitted_pipeline
        X = [(aptamer, protein) for _ in range(10)]

        preds = pipeline.predict(X)

        assert set(preds).issubset({0, 1})

    def test_predict_unfitted_raises(self):
        """Check that predict raises when not fitted."""
        pipeline = AptaMCTSPipeline()

        with pytest.raises(Exception):
            pipeline.predict([("ACGU", "ACDEF")])


# ---------------------------------------------------------------------------
# Pipeline recommend() tests
# ---------------------------------------------------------------------------


class TestAptaMCTSPipelineRecommend:
    """Tests for AptaMCTSPipeline.recommend()."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline for testing."""
        aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
        protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
        X = [(aptamer, protein) for _ in range(40)]
        y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

        pipeline = AptaMCTSPipeline(depth=5, n_iterations=2)
        pipeline.fit(X, y)
        return pipeline

    def test_recommend_returns_set(self, fitted_pipeline, monkeypatch):
        """Check that recommend returns a set of candidates."""
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

        candidates = fitted_pipeline.recommend(target="ACDEFGHIK", n_candidates=3)

        assert isinstance(candidates, set)

    def test_recommend_correct_number_of_candidates(self, fitted_pipeline, monkeypatch):
        """Check that recommend returns exactly n_candidates unique results."""
        counter = 0

        class MockMCTS:
            def __init__(self, **kwargs):
                pass

            def run(self, verbose=False):
                nonlocal counter
                counter += 1
                return {
                    "candidate": f"APT{counter:03d}",
                    "sequence": f"APT{counter:03d}_SEQ",
                    "score": np.float64(0.5),
                }

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        for n in [1, 3, 5]:
            counter = 0
            candidates = fitted_pipeline.recommend(target="ACDEFGHIK", n_candidates=n)
            assert len(candidates) == n

    def test_recommend_tuple_structure(self, fitted_pipeline, monkeypatch):
        """Check that each candidate is a (candidate, sequence, score) tuple."""
        counter = 0

        class MockMCTS:
            def __init__(self, **kwargs):
                pass

            def run(self, verbose=False):
                nonlocal counter
                counter += 1
                return {
                    "candidate": f"APT{counter:03d}",
                    "sequence": f"APT{counter:03d}_SEQ",
                    "score": np.float64(0.5),
                }

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        candidates = fitted_pipeline.recommend(target="ACDEFGHIK", n_candidates=3)

        for candidate, sequence, score in candidates:
            assert isinstance(candidate, str)
            assert isinstance(sequence, str)
            assert isinstance(score, float)

    def test_recommend_uses_mcts(self, fitted_pipeline, monkeypatch):
        """Check that recommend actually calls MCTS with correct parameters."""
        mcts_calls = []

        class MockMCTS:
            def __init__(self, experiment, depth, n_iterations):
                mcts_calls.append(
                    {"depth": depth, "n_iterations": n_iterations}
                )
                self.counter = 0

            def run(self, verbose=False):
                return {
                    "candidate": "APT000",
                    "sequence": "APT000_SEQ",
                    "score": np.float64(0.5),
                }

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        fitted_pipeline.recommend(target="ACDEF", n_candidates=1)

        assert len(mcts_calls) >= 1
        assert mcts_calls[0]["depth"] == 5
        assert mcts_calls[0]["n_iterations"] == 2

    def test_recommend_uses_experiment_adapter(self, fitted_pipeline, monkeypatch):
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
                pass

            def run(self, verbose=False):
                return {
                    "candidate": "APT000",
                    "sequence": "APT000_SEQ",
                    "score": np.float64(0.5),
                }

        monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

        fitted_pipeline.recommend(target="TESTTARGET", n_candidates=1)

        assert len(experiment_created) == 1
        assert experiment_created[0]["target"] == "TESTTARGET"

    def test_recommend_unfitted_raises(self):
        """Check that recommend raises when pipeline is not fitted."""
        pipeline = AptaMCTSPipeline()

        with pytest.raises(Exception):
            pipeline.recommend(target="ACDEF", n_candidates=1)

    def test_recommend_score_converted_to_float(self, fitted_pipeline, monkeypatch):
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

        candidates = fitted_pipeline.recommend(target="ACDEF", n_candidates=1)

        for _, _, score in candidates:
            assert type(score) is float


# ---------------------------------------------------------------------------
# Pipeline initialization tests
# ---------------------------------------------------------------------------


class TestAptaMCTSPipelineInit:
    """Tests for AptaMCTSPipeline initialization."""

    def test_default_parameters(self):
        """Check default depth and n_iterations values."""
        pipeline = AptaMCTSPipeline()

        assert pipeline.depth == 20
        assert pipeline.n_iterations == 1000
        assert pipeline.estimator is None

    @pytest.mark.parametrize("depth, n_iterations", [(5, 100), (10, 500), (50, 2000)])
    def test_custom_parameters(self, depth, n_iterations):
        """Check custom depth and n_iterations are stored correctly."""
        pipeline = AptaMCTSPipeline(depth=depth, n_iterations=n_iterations)

        assert pipeline.depth == depth
        assert pipeline.n_iterations == n_iterations
