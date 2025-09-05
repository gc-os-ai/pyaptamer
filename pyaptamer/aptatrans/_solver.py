__author__ = ["nennomp"]
__all__ = ["AptaTransSolver"]

import torch
import torch.nn as nn

from pyaptamer.base._base_solver import BaseSolver


class AptaTransSolver(BaseSolver):
    # TODO: Refactor to Pytorch Lightning if possible
    """
    Solver for AptaTrans models using binary cross-entropy loss.

    This solver is specifically designed for binary classification tasks
    with AptaTrans models, using BCELoss and computing classification accuracy.

    Parameters
    ----------
    criterion : nn.Module, optional, default=None
        Loss function to use. If None, defaults to `nn.BCELoss`.
    optimizer : type[Optimizer], optional, default=None
        Optimizer class to use. If None, defaults to `torch.optim.AdamW`.
    lr : float, optional, default=1e-5
        Learning rate for the optimizer.
    weight_decay : float, optional, default=1e-5
        Weight decay (L2 penalty) for the optimizer.
    momentum : float, optional, default=0.9
        Momentum factor for optimizers that support it. Only used if the optimizer
        supports momentum (i.e., SGD).
    betas : tuple[float, float], optional, default=(0.9, 0.999)
        Coefficients used for computing running averages of gradient and its square.
        Only used if the optimizer supports these parameters (i.e., Adam, AdamW).
    **kwargs
        Additional arguments passed to `BaseSolver.__init__`.
    """

    def __init__(
        self,
        criterion=None,
        optimizer=None,
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        momentum: float | None = 0.9,
        betas: tuple[float, float] | None = (0.9, 0.999),
        **kwargs,
    ) -> None:
        if criterion is None:
            criterion = nn.BCELoss()
        if optimizer is None:
            optimizer = torch.optim.AdamW

        super().__init__(
            optimizer=optimizer,
            criterion=criterion,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            **kwargs,
        )

    def _compute_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute binary classification accuracy.

        Parameters
        ----------
        outputs : torch.Tensor
            Model outputs (sigmoid probabilities for binary classification).
        targets : torch.Tensor
            Ground truth binary labels (0 or 1).

        Returns
        -------
        float
            Classification accuracy as a float between 0 and 1.
        """
        # convert probabilities to binary predictions using 0.5 threshold
        predictions = (outputs >= 0.5).float()
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()

        return accuracy
