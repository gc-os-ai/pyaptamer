__author__ = ["nennomp"]
__all__ = ["AptaTransSolver"]

import torch.nn as nn

from pyaptamer.training import Solver


class AptaTransSolver(Solver):
    def __init__(
        self,
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        self.criterion = nn.BCELoss()