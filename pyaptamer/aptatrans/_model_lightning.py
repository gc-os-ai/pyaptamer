"""AptaTrans' deep neural network wrapper fro Lightning."""

__author__ = ["nennomp"]
__all__ = ["AptaTransLightning", "AptaTransEncoderLightning"]


import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AptaTransLightning(L.LightningModule):
    """LightningModule wrapper for the AptaTrans deep neural network [1]_.

    This class defines a LightningModule which acts as a wrapper for the AptaTrans
    model, implemented as a `torch.nn.Module` in `pyaptamer.aptatrans._model.py`.

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
    >>> trainer.test(model_lightning, train_dataloader)  # doctest: +SKIP
    >>> preds = trainer.predict(model_lightning, train_dataloader)  # doctest: +SKIP
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

    def _step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Defines a single (mini-batch) step in the training/test/prediction loop.

        Parameters
        ----------
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]
            A batch of data containing aptamer sequences, protein sequences, and
            optionally labels.
        batch_idx: int
            Index of the batch.

        Returns
        -------
        tuple[Tensor, Tensor | None, Tensor | None]
            A tuple containing (predictions or probabilities, loss, accuracy). Loss and
            accuracy are None if ground-truth labels are not provided.
        """
        y, loss, accuracy = None, None, None

        if len(batch) == 3:  # contains ground-truth labels
            x_apta, x_prot, y = batch
        else:  # no ground-truth labels
            x_apta, x_prot = batch

        y_hat = torch.flatten(self.model(x_apta, x_prot))  # predicted probabilities
        y_pred = (y_hat > 0.5).float()  # classification labels

        if y is not None:
            # loss
            loss = F.binary_cross_entropy(y_hat, y.float())
            # accuracy
            accuracy = (y_pred == y.float()).float().mean()

        return y_pred, loss, accuracy

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
        _, loss, accuracy = self._step(batch, batch_idx)

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(
            "train_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True
        )

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """Defines a single (mini-batch) step in the test loop.

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
        _, loss, accuracy = self._step(batch, batch_idx)

        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def predict_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Predict labels for a single (mini-batch) step.

        Parameters
        ----------
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]
            A batch of data containing aptamer sequences and protein sequences.
            Optionally includes labels if available (ignored for prediction).
        batch_idx: int
            Index of the batch.

        Returns
        -------
        Tensor
            The predicted labels for the batch.
        """
        return self._step(batch, batch_idx)[0]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Defines the optimizer to be used during training."""
        params = [
            p
            for name, p in self.model.named_parameters()
            if "token_predictor" not in name
        ]

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        return optimizer


class AptaTransEncoderLightning(AptaTransLightning):
    """LightningModule wrapper for training the AptaTrans encoders [1]_.

    This class defines a LightningModule which acts as a wrapper for the AptaTrans
    encoders, implemented as a `torch.nn.Module` in `pyaptamer.aptatrans._model.py`.

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
    >>> trainer.test(model_lightning, train_dataloader)  # doctest: +SKIP
    >>> preds = trainer.predict(model_lightning, train_dataloader)  # doctest: +SKIP
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
    >>> trainer.test(model_lightning, train_dataloader)  # doctest: +SKIP
    >>> preds = trainer.predict(model_lightning, train_dataloader)  # doctest: +SKIP
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

    def _step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> tuple[tuple[Tensor, Tensor], Tensor | None]:
        """Defines a single (mini-batch) step in the training/test/prediction loop.

        The loss function is a weighted sum of the masked language modeling (MLM)
        loss and the secondary structure prediction (SSP) loss.

        Parameters
        ----------
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]
            A batch of data containing input sequences and optionally labels.
            If labels provided: (x_mlm, x_ssp, y_mlm, y_ssp).
            If no labels: (x_mlm, x_ssp).
        batch_idx: int
            Index of the batch.

        Returns
        -------
        tuple[tuple[Tensor, Tensor], Tensor | None]
            A tuple containing (predictions, loss). Predictions are a tuple of
            (mlm_predictions, ssp_predictions). Loss is None if labels are not provided.
        """
        y_mlm, y_ssp, loss = None, None, None

        if len(batch) == 4:  # contains ground-truth labels
            x_mlm, x_ssp, y_mlm, y_ssp = batch
        else:  # no ground-truth labels
            x_mlm, x_ssp = batch

        y_mlm_hat, y_ssp_hat = self.model.forward_encoder(
            x=(x_mlm, x_ssp), encoder_type=self.encoder_type
        )

        if y_mlm is not None and y_ssp is not None:
            loss_mlm = F.cross_entropy(y_mlm_hat.transpose(1, 2), y_mlm.long())
            loss_ssp = F.cross_entropy(y_ssp_hat.transpose(1, 2), y_ssp.long())
            loss = self.weight_mlm * loss_mlm + self.weight_ssp * loss_ssp

        return (y_mlm_hat, y_ssp_hat), loss

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Defines a single (mini-batch) step in the training loop.

        Parameters
        ----------
        batch: tuple[Tensor, Tensor, Tensor, Tensor]
            A batch of data containing input sequences and labels.
        batch_idx: int
            Index of the batch.

        Returns
        -------
        Tensor
            The computed loss for the batch.
        """
        _, loss = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Defines a single (mini-batch) step in the test loop.

        Parameters
        ----------
        batch: tuple[Tensor, Tensor, Tensor, Tensor]
            A batch of data containing input sequences and labels.
        batch_idx: int
            Index of the batch.

        Returns
        -------
        Tensor
            The computed loss for the batch.
        """
        _, loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def predict_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Predict masked tokens and secondary structures for a single (mini-batch) step.

        Parameters
        ----------
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]
            A batch of data containing input sequences. Optionally includes labels
            if available (ignored for prediction).
        batch_idx: int
            Index of the batch.

        Returns
        -------
        tuple[Tensor, Tensor]
            The predicted masked language modeling logits and secondary structure
            prediction logits for the batch.
        """
        return self._step(batch, batch_idx)[0]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Defines the optimizer to be used during training."""
        if self.encoder_type == "apta":
            params = list(self.model.encoder_apta.parameters()) + list(
                self.model.token_predictor_apta.parameters()
            )
        elif self.encoder_type == "prot":
            params = list(self.model.encoder_prot.parameters()) + list(
                self.model.token_predictor_prot.parameters()
            )

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        return optimizer
