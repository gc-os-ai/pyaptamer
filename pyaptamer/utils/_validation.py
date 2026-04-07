__author__ = ["Rusheel86"]
__all__ = [
    "validate_tensor",
    "validate_paired_tensors",
]

import torch
from torch import Tensor


def validate_tensor(x: Tensor, name: str = "tensor") -> None:
    """Validate that a tensor is not empty and contains only finite values."""
    if x.numel() == 0:
        raise ValueError(f"Input {name} cannot be empty. Got shape {tuple(x.shape)}.")

    if not torch.isfinite(x).all():
        raise ValueError(f"Input {name} contains NaN or Inf values.")


def validate_paired_tensors(x_apta: Tensor, x_prot: Tensor) -> None:
    """Validate paired aptamer and protein tensors."""
    validate_tensor(x_apta, "x_apta")
    validate_tensor(x_prot, "x_prot")

    if x_apta.shape[0] != x_prot.shape[0]:
        raise ValueError(
            f"Batch sizes must match. Got {x_apta.shape[0]} for x_apta "
            f"and {x_prot.shape[0]} for x_prot."
        )
