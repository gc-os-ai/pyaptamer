"""AptaTrans' deep neural network wrapper fro Lightning."""

__author__ = ["nennomp"]
__all__ = ["AptaTransLightning", "AptaTransEncoderLightning"]


import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AptaTransLightning(L.LightningModule):
    """LightningModule wrapper for training the AptaTrans deep neural network [1]_.

    This class defines a LightningModule which acts as a wrapper for the AptaTrans
    model, implemented as a `torch.nn.Module` in `pyaptamer.aptatrans._model.py`.
    Specifically, it implements two methods to make it compatible with lightning
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
        loss = F.binary_cross_entropy(y_hat.squeeze(0), y.float())
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


class AptaTransEncoderLightning(AptaTransLightning):
    """LightningModule wrapper for training the AptaTrans encoders [1]_.

    This class defines a LightningModule which acts as a wrapper for the AptaTrans
    encoders, implemented as a `torch.nn.Module` in `pyaptamer.aptatrans._model.py`.
    Specifically, it implements two methods to make it compatible with lightning
    training interface: (i) `training_step`, defining the training loop and (ii)
    `configure_optimizers`, defining the optimizer used for training.

    Parameters
    ----------
    model: AptaTrans
        An instance of the AptaTrans model.
    encoder_type: str
        A string indicating whether to use the aptamer or protein encoder. Options
        are 'apta' or 'prot'.
    lr: float, optional, default=1e-5
        Learning rate for the optimizer.
    weight_decay: float, optional, default=1e-5
        Weight decay (L2 regularization) for the optimizer.
    betas: tuple[float, float], optional, default=(0.9, 0.999)
        Momentum coefficients for the Adam optimizer.
    weight_mlm: float, optional, default=2.0
        Weight for the masked language modeling (MLM) loss in the weighted total loss.
    weight_ssp: float, optional, default=1.0
        Weight for the secondary structure prediction (SSP) loss in the weighted
        total loss.

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
    ...     AptaTransEncoderLightning,
    ...     EncoderPredictorConfig,
    ... )
    >>> apta_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> prot_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> model = AptaTrans(apta_embedding, prot_embedding)
    >>> # pretrain aptamer encoder
    >>> model_lightning = AptaTransEncoderLightning(model, encoder_type="apta")
    >>> x_apta_mlm = torch.randint(0, 125, (8, 128))
    >>> x_apta_ssp = torch.randint(0, 125, (8, 128))
    >>> y_mlm = torch.randint(0, 125, (8, 128))
    >>> y_ssp = torch.randint(0, 8, (8, 128))
    >>> train_dataloader = torch.utils.data.DataLoader(
    ...     list(zip(x_apta_mlm, x_apta_ssp, y_mlm, y_ssp)),
    ...     batch_size=4,
    ...     shuffle=True,
    ... )
    >>> trainer = L.Trainer(max_epochs=1)
    >>> trainer.fit(model_lightning, train_dataloader)  # doctest: +SKIP
    >>> # pretrain protein encoder
    >>> model_lightning = AptaTransEncoderLightning(model, encoder_type="prot")
    >>> x_prot_mlm = torch.randint(0, 25, (8, 128))
    >>> x_prot_ssp = torch.randint(0, 25, (8, 128))
    >>> y_mlm = torch.randint(0, 25, (8, 128))
    >>> y_ssp = torch.randint(0, 3, (8, 128))
    >>> train_dataloader = torch.utils.data.DataLoader(
    ...     list(zip(x_prot_mlm, x_prot_ssp, y_mlm, y_ssp)),
    ...     batch_size=4,
    ...     shuffle=True,
    ... )
    >>> trainer = L.Trainer(max_epochs=1)
    >>> trainer.fit(model_lightning, train_dataloader)  # doctest: +SKIP
    """

    def __init__(
        self,
        model: nn.Module,
        encoder_type: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_mlm: float = 2.0,
        weight_ssp: float = 1.0,
    ) -> None:
        super().__init__(model, lr, weight_decay, betas)
        self.encoder_type = encoder_type
        self.weight_mlm = weight_mlm
        self.weight_ssp = weight_ssp

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Defines a single (mini-batch) step in the training loop.

        The loss function is a weighted sum of the masked language modeling (MLM)
        loss and the secondary structure prediction (SSP) loss.

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
        # (input masked, secondary structure, ground-truth targets)
        x_mlm, x_ssp, y_mlm, y_ssp = batch
        y_mlm_hat, y_ssp_hat = self.model.forward_encoder(
            x=(x_mlm, x_ssp), encoder_type=self.encoder_type
        )

        loss_mlm = F.cross_entropy(y_mlm_hat, y_mlm.float())
        loss_ssp = F.cross_entropy(y_ssp_hat, y_ssp.float())
        return self.weight_mlm * loss_mlm + self.weight_ssp * loss_ssp
