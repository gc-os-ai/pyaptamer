"""AptaTrans' deep neural network wrapper fro Lightning."""

__author__ = ["nennomp"]
__all__ = ["AptaTransLightning"]


import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AptaTransLightning(L.LightningModule):
    """LightningModule wrapper for the AptaTrans deep neural network [1]_.

    This class defines a LightningModule which acts as a wrapper for the AptaTrans
    model, implemented as a `torch.nn.Module` in `pyaptamer.aptatrans._model.py`.
    Specifically, it implementa two methods to make it compatible with lightning
    training interface: (i) `training_step`, defining the training loop and (ii)
    `configure_optimizers`, defining the optimizer used for training.

    Parameters
    ----------
    model: AptaTrans
        An instance of the AptaTrans model.
    lr: float, optional, default=1e-5
        Learning rate for the optimizer.
    weight_decay: float, optional, default=1e-5
        Weight decay (L2 regularization) for the optimizer.
    betas: tuple[float, float], optional, default=(0.9, 0.999)
        Momentum coefficients for the Adam optimizer.

    References
    ----------
    .. [1] Shin, Incheol, et al. "AptaTrans: a deep neural network for predicting
    aptamer-protein interaction using pretrained encoders." BMC bioinformatics 24.1
    (2023): 447.

    Examples
    --------
    >>> import lightning as L
    >>> import torch
    >>> from pyaptamer.aptatrans import (
    ...     AptaTrans,
    ...     AptaTransLightning,
    ...     EncoderPredictorConfig,
    ... )
    >>> apta_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> prot_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> model = AptaTrans(apta_embedding, prot_embedding)
    >>> model_lightning = AptaTransLightning(model)
    >>> # dummy data
    >>> x_apta = torch.randint(0, 4, (8, 128))
    >>> x_prot = torch.randint(0, 20, (8, 128))
    >>> y = torch.randint(0, 2, (8, 1))
    >>> train_dataloader = torch.utils.data.DataLoader(
    ...     list(zip(x_apta, x_prot, y)),
    ...     batch_size=4,
    ...     shuffle=True,
    ... )
    >>> trainer = L.Trainer(max_epochs=1)
    >>> trainer.fit(model_lightning, train_dataloader)  # doctest: +SKIP
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Defines a single (mini-batch) step in the training loop.

        Parameters
        ----------
        batch: tuple[Tensor, Tensor, Tensor]
            A batch of data containing aptamer sequences, protein sequences, and labels.
        batch_idx: int
            Index of the batch.

        Returns
        -------
        Tensor
            The computed loss for the batch.
        """
        # (input aptamers, input proteins, ground-truth targets)
        x_apta, x_prot, y = batch
        y_hat = self.model(x_apta, x_prot)
        loss = F.binary_cross_entropy(y_hat, y.float())
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Defines the optimizer to be used during training."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        return optimizer
