"""Checkpoint utilities for saving and resuming MCTS runs."""

__author__ = ["github.com/ritankarsaha"]
__all__ = ["MCTSRunCheckpoint", "RecommendCheckpoint"]

import json
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class MCTSRunCheckpoint:
    """Checkpoint for a single MCTS.run() call.

    Persists the best subsequence (`base`) found after each round so
    that a long MCTS run can be resumed after a crash or interruption.

    Parameters
    path : str or Path
        File path where the checkpoint JSON is written.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, base: str, round_idx: int, depth: int) -> None:
        """Write the current run state to disk.

        Parameters
        base : str
            Best encoded subsequence found so far.
        round_idx : int
            Index of the last completed round (0-based).
        depth : int
            Target sequence depth for this run.
        """
        data = {
            "base": base,
            "round": round_idx,
            "depth": depth,
            "timestamp": _now_iso(),
        }
        self.path.write_text(json.dumps(data, indent=2))

    def load(self) -> dict | None:
        """Load a previously saved checkpoint from disk.

        Returns
        dict or None
            The checkpoint data, or None if the file does not exist or
            is unreadable.
        """
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self) -> None:
        """Delete the checkpoint file."""
        if self.path.exists():
            self.path.unlink()


class RecommendCheckpoint:
    """Checkpoint for AptaTransPipeline.recommend().

    Persists discovered candidate aptamers after each new unique
    candidate is found, allowing long recommendation runs to resume
    after a crash or interruption.

    Parameters
    path : str or Path
        File path where the checkpoint JSON is written.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(
        self,
        target: str,
        n_candidates: int,
        depth: int,
        n_iterations: int,
        candidates: dict,
        completed: bool = False,
    ) -> None:
        """Write the current recommendation state to disk.

        Parameters
        target : str
            The target protein sequence.
        n_candidates : int
            The total number of unique candidates requested.
        depth : int
            MCTS depth used for this run.
        n_iterations : int
            MCTS iterations per round used for this run.
        candidates : dict
            Currently discovered candidates, keyed by reconstructed
            sequence. Each value is a list [candidate, sequence, score].
        completed : bool, optional, default=False
            Whether the recommendation run completed successfully.
        """
        data = {
            "target": target,
            "n_candidates": n_candidates,
            "depth": depth,
            "n_iterations": n_iterations,
            "candidates": candidates,
            "completed": completed,
            "timestamp": _now_iso(),
        }
        self.path.write_text(json.dumps(data, indent=2))

    def load(self) -> dict | None:
        """Load a previously saved checkpoint from disk.

        Returns
        dict or None
            The checkpoint data, or None if the file does not exist or
            is unreadable.
        """
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self) -> None:
        """Delete the checkpoint file."""
        if self.path.exists():
            self.path.unlink()
