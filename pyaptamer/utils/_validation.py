__author__ = ["nennomp", "satvshr"]
__all__ = [
    "validate_aptanet_mlp_input",
    "validate_aptatrans_inputs",
    "validate_interaction_map_inputs",
]

import torch
from torch import Tensor


def _validate_tensor(
    x: Tensor,
    *,
    name: str,
    expected_ndim: int,
    check_finite: bool,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor, got {type(x).__name__}.")

    if x.ndim != expected_ndim:
        raise ValueError(
            f"`{name}` must be a {expected_ndim}D tensor, got shape {tuple(x.shape)}."
        )

    if x.numel() == 0 or any(dim == 0 for dim in x.shape):
        raise ValueError(f"`{name}` must be non-empty, got shape {tuple(x.shape)}.")

    if check_finite and x.is_floating_point() and not torch.isfinite(x).all():
        raise ValueError(f"`{name}` contains NaN or Inf values.")


def validate_aptatrans_inputs(x_apta: Tensor, x_prot: Tensor) -> None:
    """Validate input tensors used by AptaTrans model forward passes."""
    _validate_tensor(
        x_apta,
        name="x_apta",
        expected_ndim=2,
        check_finite=True,
    )
    _validate_tensor(
        x_prot,
        name="x_prot",
        expected_ndim=2,
        check_finite=True,
    )

    if x_apta.shape[0] != x_prot.shape[0]:
        raise ValueError(
            "`x_apta` and `x_prot` must have the same batch size, "
            f"got {x_apta.shape[0]} and {x_prot.shape[0]}."
        )


def validate_interaction_map_inputs(x_apta: Tensor, x_prot: Tensor) -> None:
    """Validate encoded sequence tensors used by InteractionMap."""
    _validate_tensor(
        x_apta,
        name="x_apta",
        expected_ndim=3,
        check_finite=True,
    )
    _validate_tensor(
        x_prot,
        name="x_prot",
        expected_ndim=3,
        check_finite=True,
    )

    if x_apta.shape[0] != x_prot.shape[0]:
        raise ValueError(
            "`x_apta` and `x_prot` must have the same batch size, "
            f"got {x_apta.shape[0]} and {x_prot.shape[0]}."
        )

    if x_apta.shape[-1] != x_prot.shape[-1]:
        raise ValueError(
            "The number of features of `x_apta` and `x_prot` must match, "
            f"got {x_apta.shape[-1]} and {x_prot.shape[-1]}."
        )


def validate_aptanet_mlp_input(x: Tensor) -> None:
    """Validate input tensor used by AptaNetMLP forward pass."""
    _validate_tensor(
        x,
        name="x",
        expected_ndim=2,
        check_finite=True,
    )
