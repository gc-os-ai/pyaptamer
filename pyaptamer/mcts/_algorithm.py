"""Monte Carlo Tree Search (MCTS) for string optimization."""

__author__ = ["nennomp"]
__all__ = ["MCTS"]


from abc import ABC, abstractmethod


class MCTS(ABC):
    """Abstract base class for MCTS algorithm implementation."""
    
    def __init__(
        self,
        experiment,
        states: list[str] | None = None,
        depth: int = 20,
        n_iterations: int = 1000,
    ) -> None:
        """Initialize MCTS with basic parameters."""
        pass
    
    @abstractmethod
    def run(self, verbose: bool = True) -> dict:
        pass
    
    @abstractmethod
    def _reset(self) -> None:
        pass
    
    @abstractmethod
    def _selection(self, node) -> object:
        pass
    
    @abstractmethod
    def _expansion(self, node) -> object:
        pass
    
    @abstractmethod
    def _simulation(self, node) -> float:
        pass
    
    @abstractmethod 
    def _find_best_subsequence(self) -> str:
        pass