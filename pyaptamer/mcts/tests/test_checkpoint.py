"""Test suite for MCTS checkpoint utilities."""

__author__ = ["github.com/ritankarsaha"]

import json

import numpy as np
import pytest
import torch

from pyaptamer.experiments import AptamerEvalAptaNet, AptamerEvalAptaTrans
from pyaptamer.mcts import MCTS, MCTSRunCheckpoint, RecommendCheckpoint

NUCLEOTIDES = ["A_", "_A", "C_", "_C", "G_", "_G", "U_", "_U"]


class MockModel:
    def __init__(self):
        self.apta_embedding = type("obj", (object,), {"max_len": 128})
        self.prot_embedding = type("obj", (object,), {"max_len": 128})

    def eval(self):
        pass

    def __call__(self, *args, **kwargs):
        return torch.tensor([0.5])


class MockPipeline:
    def predict_proba(self, X):
        return np.array([[0.5, 0.5]])


class MockExperimentAptaTrans(AptamerEvalAptaTrans):
    def __init__(self, target, model, device, prot_words, fixed_score=0.5):
        super().__init__(target, model, device, prot_words)
        self.fixed_score = fixed_score

    def evaluate(self, aptamer_candidate):
        return torch.tensor([self.fixed_score])


class MockExperimentAptaNet(AptamerEvalAptaNet):
    def __init__(self, target, pipeline, fixed_score=0.5):
        super().__init__(target, pipeline)
        self.fixed_score = fixed_score

    def evaluate(self, aptamer_candidate):
        return np.float64(self.fixed_score)


@pytest.fixture(params=["aptatrans", "aptanet"])
def mcts(request):
    if request.param == "aptatrans":
        mock_model = MockModel()
        device = torch.device("cpu")
        prot_words = {"AAA": 0.5, "AAC": 0.3, "AAG": 0.2}
        experiment = MockExperimentAptaTrans(
            target="ACGU", model=mock_model, device=device, prot_words=prot_words
        )
    else:
        mock_pipeline = MockPipeline()
        experiment = MockExperimentAptaNet(target="ACGU", pipeline=mock_pipeline)

    return MCTS(experiment=experiment, states=NUCLEOTIDES, depth=5, n_iterations=10)


# MCTSRunCheckpoint tests


class TestMCTSRunCheckpoint:
    """Tests for MCTSRunCheckpoint."""

    def test_save_creates_file(self, tmp_path):
        """Saving should create the checkpoint file."""
        path = tmp_path / "run.json"
        cp = MCTSRunCheckpoint(path)
        cp.save(base="A_C_", round_idx=1, depth=5)
        assert path.exists()

    def test_save_content(self, tmp_path):
        """Saved file should contain the correct fields."""
        path = tmp_path / "run.json"
        cp = MCTSRunCheckpoint(path)
        cp.save(base="A_C_", round_idx=2, depth=5)
        data = json.loads(path.read_text())
        assert data["base"] == "A_C_"
        assert data["round"] == 2
        assert data["depth"] == 5
        assert "timestamp" in data

    def test_load_existing(self, tmp_path):
        """Loading an existing checkpoint returns the stored dict."""
        path = tmp_path / "run.json"
        cp = MCTSRunCheckpoint(path)
        cp.save(base="G_U_", round_idx=3, depth=5)
        loaded = cp.load()
        assert loaded is not None
        assert loaded["base"] == "G_U_"
        assert loaded["round"] == 3

    def test_load_missing_returns_none(self, tmp_path):
        """Loading from a non-existent path returns None."""
        cp = MCTSRunCheckpoint(tmp_path / "nonexistent.json")
        assert cp.load() is None

    def test_load_corrupted_returns_none(self, tmp_path):
        """Loading a corrupted JSON file returns None without raising."""
        path = tmp_path / "bad.json"
        path.write_text("{ not valid json }")
        cp = MCTSRunCheckpoint(path)
        assert cp.load() is None

    def test_clear_removes_file(self, tmp_path):
        """Clearing should delete the checkpoint file."""
        path = tmp_path / "run.json"
        cp = MCTSRunCheckpoint(path)
        cp.save(base="A_", round_idx=0, depth=5)
        assert path.exists()
        cp.clear()
        assert not path.exists()

    def test_clear_no_file_is_noop(self, tmp_path):
        """Clearing when no file exists should not raise."""
        cp = MCTSRunCheckpoint(tmp_path / "gone.json")
        cp.clear()  # should not raise


# RecommendCheckpoint tests


class TestRecommendCheckpoint:
    """Tests for RecommendCheckpoint."""

    def _sample_candidates(self):
        return {
            "ACGUA": ["ACGUA", "A_C_G_U_A_", 0.85],
            "UACGU": ["UACGU", "_U_A_C_G_U", 0.72],
        }

    def test_save_creates_file(self, tmp_path):
        """Saving should create the checkpoint file."""
        path = tmp_path / "rec.json"
        cp = RecommendCheckpoint(path)
        cp.save(
            target="MSEQ",
            n_candidates=5,
            depth=20,
            n_iterations=1000,
            candidates=self._sample_candidates(),
        )
        assert path.exists()

    def test_save_content(self, tmp_path):
        """Saved file should contain the correct top-level fields."""
        path = tmp_path / "rec.json"
        cp = RecommendCheckpoint(path)
        candidates = self._sample_candidates()
        cp.save(
            target="MSEQ",
            n_candidates=5,
            depth=20,
            n_iterations=1000,
            candidates=candidates,
            completed=False,
        )
        data = json.loads(path.read_text())
        assert data["target"] == "MSEQ"
        assert data["n_candidates"] == 5
        assert data["depth"] == 20
        assert data["n_iterations"] == 1000
        assert data["completed"] is False
        assert set(data["candidates"].keys()) == set(candidates.keys())

    def test_save_completed_flag(self, tmp_path):
        """completed flag should be stored correctly."""
        path = tmp_path / "rec.json"
        cp = RecommendCheckpoint(path)
        cp.save(
            target="X",
            n_candidates=1,
            depth=5,
            n_iterations=10,
            candidates={},
            completed=True,
        )
        data = json.loads(path.read_text())
        assert data["completed"] is True

    def test_load_existing(self, tmp_path):
        """Loading an existing checkpoint returns the stored dict."""
        path = tmp_path / "rec.json"
        cp = RecommendCheckpoint(path)
        candidates = self._sample_candidates()
        cp.save(
            target="MSEQ",
            n_candidates=5,
            depth=20,
            n_iterations=1000,
            candidates=candidates,
        )
        loaded = cp.load()
        assert loaded is not None
        assert loaded["target"] == "MSEQ"
        assert set(loaded["candidates"].keys()) == set(candidates.keys())

    def test_load_missing_returns_none(self, tmp_path):
        """Loading from a non-existent path returns None."""
        cp = RecommendCheckpoint(tmp_path / "no_such_file.json")
        assert cp.load() is None

    def test_load_corrupted_returns_none(self, tmp_path):
        """Loading a corrupted JSON file returns None without raising."""
        path = tmp_path / "bad.json"
        path.write_text("definitely not json !!!")
        cp = RecommendCheckpoint(path)
        assert cp.load() is None

    def test_clear_removes_file(self, tmp_path):
        """Clearing should delete the checkpoint file."""
        path = tmp_path / "rec.json"
        cp = RecommendCheckpoint(path)
        cp.save(
            target="X",
            n_candidates=1,
            depth=5,
            n_iterations=10,
            candidates={},
        )
        cp.clear()
        assert not path.exists()

    def test_clear_no_file_is_noop(self, tmp_path):
        """Clearing when no file exists should not raise."""
        cp = RecommendCheckpoint(tmp_path / "gone.json")
        cp.clear()


# Integration: MCTS.run() with checkpoint_path


class TestMCTSRunCheckpointing:
    """Integration tests for MCTS.run() checkpointing."""

    def test_run_creates_checkpoint_during_run(self, mcts, tmp_path):
        """Checkpoint file should exist after run() completes and then be cleared."""
        path = tmp_path / "mcts_run.json"
        mcts.run(verbose=False, checkpoint_path=path)
        # checkpoint is deleted on clean completion
        assert not path.exists()

    def test_run_without_checkpoint_path(self, mcts, tmp_path):
        """run() without checkpoint_path should complete normally."""
        result = mcts.run(verbose=False)
        assert isinstance(result, dict)
        assert "candidate" in result

    def test_run_resumes_from_complete_base(self, mcts, tmp_path):
        """A checkpoint with a fully built base should skip all rounds."""
        path = tmp_path / "mcts_run.json"
        cp = MCTSRunCheckpoint(path)

        # depth=5 -> full base has 10 chars; pre-seed a complete one
        full_base = "A_C_G_U_A_"
        assert len(full_base) == mcts.depth * 2
        cp.save(base=full_base, round_idx=4, depth=mcts.depth)

        result = mcts.run(verbose=False, checkpoint_path=path)

        # the reconstructed sequence should match the saved base
        assert result["sequence"] == full_base
        assert len(result["candidate"]) == mcts.depth
        assert isinstance(result["score"], torch.Tensor | np.floating | float)

    def test_run_ignores_incompatible_checkpoint(self, mcts, tmp_path):
        """A checkpoint with a different depth should be ignored."""
        path = tmp_path / "mcts_run.json"
        cp = MCTSRunCheckpoint(path)
        # depth mismatch: checkpoint says depth=99, mcts has depth=5
        cp.save(base="A_C_G_U_A_", round_idx=4, depth=99)

        # should run from scratch and still complete
        result = mcts.run(verbose=False, checkpoint_path=path)
        assert isinstance(result, dict)
        assert len(result["candidate"]) == mcts.depth

    def test_run_ignores_missing_checkpoint(self, mcts, tmp_path):
        """A non-existent checkpoint path should not raise and run normally."""
        path = tmp_path / "does_not_exist.json"
        result = mcts.run(verbose=False, checkpoint_path=path)
        assert isinstance(result, dict)
        assert len(result["candidate"]) == mcts.depth

    def test_run_writes_checkpoint_after_each_round(self, mcts, tmp_path, monkeypatch):
        """Checkpoint file is updated after every completed round."""
        path = tmp_path / "mcts_run.json"
        saved_rounds = []

        original_save = MCTSRunCheckpoint.save

        def tracking_save(self_cp, base, round_idx, depth):
            saved_rounds.append(round_idx)
            original_save(self_cp, base=base, round_idx=round_idx, depth=depth)

        monkeypatch.setattr(MCTSRunCheckpoint, "save", tracking_save)

        mcts.run(verbose=False, checkpoint_path=path)

        # one save per round; depth=5 means 5 rounds
        assert len(saved_rounds) == mcts.depth
        assert saved_rounds == list(range(mcts.depth))
