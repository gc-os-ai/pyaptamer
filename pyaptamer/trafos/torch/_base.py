"""Base class for PyTorch-compatible transformations."""

from abc import ABC, abstractmethod

from torch import Tensor


class BaseTorchTransform(ABC):
    """Base class for torch-compatible transformations.

    Subclasses must implement ``__call__``.
    """

    @abstractmethod
    def __call__(self, x: str | Tensor) -> str | Tensor:
        """Apply transformation."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
