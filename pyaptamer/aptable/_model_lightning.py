"""Lightning wrapper for AptaBLE training."""

__author__ = ["DZDasherKTB"]
__all__ = ["AptaBLELightning"]

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pyaptamer.aptable._model import AptaBLE


class AptaBLELightning(L.LightningModule):
    """LightningModule wrapper for training AptaBLE end-to-end.

    Identical training interface to ``AptaTransLightning`` so existing
    training scripts require no changes beyond swapping the model class.

    Parameters
    ----------
    model : AptaBLE
        An instance of the AptaBLE model.
    lr : float, optional, default=1e-5
        Learning rate.
    weight_decay : float, optional, default=1e-5
        L2 regularisation weight.
    betas : tuple[float, float], optional, default=(0.9, 0.999)
        Adam momentum coefficients.

    Examples
    --------
    >>> import torch, lightning as L
    >>> from pyaptamer.aptable import AptaBLE, AptaBLELightning
    >>> from pyaptamer.aptatrans import EncoderPredictorConfig
    >>> cfg_a = EncoderPredictorConfig(128, 16, max_len=128)
    >>> cfg_p = EncoderPredictorConfig(128, 16, max_len=128)
    >>> model = AptaBLE(cfg_a, cfg_p, in_dim=32, n_encoder_layers=1,
    ...                 n_heads=4, cross_attention_heads=4, conv_layers=[1,1,1])
    >>> lit = AptaBLELightning(model)
    """

    def __init__(
        self,
        model: AptaBLE,
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def _log_metric(self, name: str, value: Tensor) -> None:
        self.log(name, value, on_epoch=True, on_step=False, prog_bar=True)

    def _step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int, stage: str
    ) -> Tensor:
        x_apta, x_prot, y = batch
        y_hat = torch.flatten(self.model(x_apta, x_prot))
        loss = F.binary_cross_entropy(y_hat, y.float())

        y_pred = (y_hat > 0.5).float()
        accuracy = (y_pred == y.float()).float().mean()

        self._log_metric(f"{stage}_loss", loss)
        self._log_metric(f"{stage}_accuracy", accuracy)
        return loss

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        return self._step(batch, batch_idx, "val")

    def test_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam optimizer excluding token predictor parameters.

        Token predictors are only used during encoder pretraining.
        Fine-tuning AptaBLE end-to-end excludes them from the optimizer,
        consistent with AptaTransLightning.
        """
        params = [
            p for name, p in self.model.named_parameters()
            if "token_predictor" not in name
        ]
        return torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
