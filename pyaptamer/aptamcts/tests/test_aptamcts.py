"""Test suite for AptaMCTS pipeline."""

__author__ = ["codex"]

import numpy as np

from pyaptamer.aptamcts import AptaMCTS


class MockAptaNetPipeline:
    """Mock AptaNet pipeline for deterministic score assignment."""

    def __init__(self, fixed_score=0.7):
        self.fixed_score = fixed_score

    def predict_proba(self, X):
        return np.array([[1 - self.fixed_score, self.fixed_score]] * len(X))


def test_predict():
    """Check `predict` returns scalar score from AptaNet experiment."""
    pipeline = MockAptaNetPipeline(fixed_score=0.8)
    aptamcts = AptaMCTS(pipeline=pipeline, depth=5, n_iterations=3)

    score = aptamcts.predict(candidate="AUGCA", target="ACDEFGHIK")
    assert isinstance(score, np.float64)
    assert score == np.float64(0.8)


def test_recommend(monkeypatch):
    """Check `recommend` returns requested number of unique candidates."""
    pipeline = MockAptaNetPipeline(fixed_score=0.6)
    aptamcts = AptaMCTS(pipeline=pipeline, depth=5, n_iterations=3)

    class MockMCTS:
        def __init__(self, **kwargs):
            self.counter = 0

        def run(self, verbose: bool = False):
            candidate_data = [
                ("AAAAA", "A_A_A_A_A_", np.float64(0.91)),
                ("CCCCC", "C_C_C_C_C_", np.float64(0.81)),
                ("GGGGG", "G_G_G_G_G_", np.float64(0.71)),
            ]
            data = candidate_data[self.counter % len(candidate_data)]
            self.counter += 1
            return {
                "candidate": data[0],
                "sequence": data[1],
                "score": data[2],
            }

    monkeypatch.setattr("pyaptamer.aptamcts._pipeline.MCTS", MockMCTS)

    candidates = aptamcts.recommend(target="ACDEFGHIK", n_candidates=2, verbose=False)
    assert isinstance(candidates, set)
    assert len(candidates) == 2

    for candidate, sequence, score in candidates:
        assert isinstance(candidate, str)
        assert isinstance(sequence, str)
        assert isinstance(score, np.float64)
