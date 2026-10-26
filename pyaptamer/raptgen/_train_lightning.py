"""LightningModule defining the training steps for RaptGen."""

__author__ = ["NoorMajdoub", "siddharth7113"]
__all__ = ["RaptGenLightning"]

from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from torch import Tensor


class RaptGenLightning(L.LightningModule):
    """LightningModule wrapper for the RaptGen variational autoencoder [1]_.

    This class defines a LightningModule which acts as a wrapper for a RaptGen
    model, `CNNPHMMVAE` or `CNNPHMMVAEFast`, implemented as a `torch.nn.Module`
    in `pyaptamer.raptgen._model.py`. It implements three methods to make the
    model compatible with the lightning training interface: (i) `training_step`,
    computing the loss of one training batch, (ii) `validation_step`, computing
    the loss of one validation batch and (iii) `configure_optimizers`, defining
    the optimizer used for training.

    The training loss is the profile HMM reconstruction loss plus `beta` times
    the KL divergence. Following the original implementation, `beta` can be
    increased linearly from zero during the first epochs, and a force-matching
    penalty can be added during the first epochs to push the decoder towards
    match states. The validation loss is the reconstruction loss plus the
    unweighted KL divergence.

    Parameters
    ----------
    model: CNNPHMMVAE or CNNPHMMVAEFast
        An instance of the RaptGen model.
    optimizer: callable, optional, default=torch.optim.Adam
        Optimizer class. It is called with the model parameters, `lr` and
        `optimizer_kwargs`.
    optimizer_kwargs: dict, optional, default=None
        Additional keyword arguments for the optimizer.
    lr: float, optional, default=1e-3
        Learning rate for the optimizer.
    beta: float, optional, default=1.0
        Weight of the KL divergence in the training loss.
    beta_schedule: bool, optional, default=True
        If True, the KL weight grows linearly from zero to `beta` during the
        first `beta_epochs` epochs.
    beta_epochs: int, optional, default=50
        Number of epochs over which the KL weight grows.
    force_matching: bool, optional, default=True
        If True, the force-matching penalty is added to the training loss
        during the first `force_epochs` epochs. Its cost decreases linearly
        from 5 to 1.
    force_epochs: int, optional, default=50
        Number of epochs during which the force-matching penalty is applied.

    References
    ----------
    .. [1] Iwano, Natsuki, et al. "Generative aptamer discovery using RaptGen."
    Nature Computational Science 2.6 (2022): 378-386.

    Examples
    --------
    >>> import lightning as L
    >>> import torch
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> from pyaptamer.raptgen import CNNPHMMVAE, RaptGenLightning
    >>> model = CNNPHMMVAE(motif_len=5)
    >>> model_lightning = RaptGenLightning(model)
    >>> # dummy data
    >>> x = torch.randint(0, 4, (8, 12))
    >>> train_dataloader = DataLoader(TensorDataset(x), batch_size=4, shuffle=True)
    >>> trainer = L.Trainer(max_epochs=1)
    >>> trainer.fit(model_lightning, train_dataloader)  # doctest: +SKIP
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr: float = 1e-3,
        beta: float = 1.0,
        beta_schedule: bool = True,
        beta_epochs: int = 50,
        force_matching: bool = True,
        force_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.beta_epochs = beta_epochs
        self.force_matching = force_matching
        self.force_epochs = force_epochs

    def _epoch(self) -> int:
        """Returns the current epoch, counted from 1 as in the original code."""
        return self.current_epoch + 1

    def _beta_at(self, epoch: int) -> float:
        """Returns the KL weight for the given epoch, counted from 1."""
        if self.beta_schedule and epoch < self.beta_epochs:
            return self.beta * epoch / self.beta_epochs
        return self.beta

    def training_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        """Defines a single (mini-batch) step in the training loop.

        Parameters
        ----------
        batch: tuple[Tensor]
            A batch of data containing the integer-encoded sequences, of shape
            (batch_size, seq_len).
        batch_idx: int
            Index of the batch.

        Returns
        -------
        Tensor
            The computed loss for the batch.
        """
        data = batch[0]
        epoch = self._epoch()
        kwargs: dict[str, Any] = {"beta": self._beta_at(epoch)}
        if self.force_matching and epoch <= self.force_epochs:
            kwargs["force_matching"] = True
            kwargs["match_cost"] = 1 + 4 * (1 - epoch / self.force_epochs)
        loss = self.model.loss_fn(data, *self.model(data), **kwargs)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=data.size(0),
        )
        return loss

    def validation_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        """Defines a single (mini-batch) step in the validation loop.

        Parameters
        ----------
        batch: tuple[Tensor]
            A batch of data containing the integer-encoded sequences, of shape
            (batch_size, seq_len).
        batch_idx: int
            Index of the batch.

        Returns
        -------
        Tensor
            The reconstruction loss plus the KL divergence for the batch.
        """
        data = batch[0]
        ce, kld = self.model.loss_fn(data, *self.model(data), test=True)
        loss = torch.tensor(ce + kld, device=data.device)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=data.size(0),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Defines the optimizer to be used during training."""
        return self.optimizer(
            self.model.parameters(), lr=self.lr, **(self.optimizer_kwargs or {})
        )
