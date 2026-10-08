"""LightningModule defining the training steps for AptaDiff."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiffLightning"]

import math
from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from torch import Tensor


class AptaDiffLightning(L.LightningModule):
    """LightningModule defining the training steps for AptaDiff.

    Wraps an `AptaDiffDiffusion` instance so it can be trained with a Lightning
    `Trainer`. This class handles only the per-batch loss and the optimizer
    configuration. The training loop, device, gradient accumulation
    and checkpointing belong to the `Trainer`, and are configured by
    :class:`~pyaptamer.aptadiff.AptaDiffGenerator`.

    The loss is the diffusion model's variational bound in bits per character
    (BPC), as reported by the original implementation. It is logged as `train_loss`
    and `val_loss`.

    Parameters
    ----------
    diffusion : AptaDiffDiffusion
        The core diffusion model to train. It has a `log_prob(x, z)` that returns
        a loss. Once it is passed here, Lightning will automatically move its weights
        to the configured device during training.
    optimizer : callable, optional, default=torch.optim.Adam
        Optimizer class, e.g. `torch.optim.AdamW` or `torch.optim.SGD`, or any
        callable that builds an optimizer from the model parameters and `lr`. The
        default is the optimizer the original implementation trained with. The
        optimizer must be compatible with dense gradients and parameters of any
        number of dimensions.
    optimizer_kwargs : dict, optional, default=None
        Keyword arguments to be passed to the optimizer, for example
        ``{"momentum": 0.9}`` for `torch.optim.SGD` or ``{"weight_decay": 0.01}``
        for `torch.optim.AdamW`. The optimizer checks them when it is built, at the
        start of training. This must not contain `lr` or `params`. PyTorch's
        defaults will be applied to anything omitted. With the default
        `torch.optim.Adam`, `optimizer_kwargs=None` reproduces the original
        configuration exactly.
    lr : float, optional, default=1e-4
        Initial learning rate, as used by the original implementation.
        Overrides PyTorch's default learning rate for the chosen optimizer.
    gamma : float, optional, default=0.99
        Per-epoch decay factor of the `ExponentialLR` schedule. Must lie
        in (0.0, 1.0]. Value 1.0 keeps the learning rate constant. Because the decay
        is exponential, values below 1.0 shrink the learning rate quickly over long
        runs, so later epochs update the model very little. Raise gamma towards 1.0
        to train longer at a useful rate.

    Raises
    ------
    ValueError
        If `gamma` lies outside (0.0, 1.0].

    References
    ----------
    .. [1] Wang, Z., et al. "AptaDiff: de novo design and optimization of
           aptamers based on diffusion models." Briefings in Bioinformatics,
           25(6), bbae517 (2024). https://doi.org/10.1093/bib/bbae517

           Original implementation: https://github.com/wz-create/AptaDiff

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.aptadiff import (
    ...     AptaDiffDenoiser,
    ...     AptaDiffDiffusion,
    ...     AptaDiffLightning,
    ... )
    >>> denoiser = AptaDiffDenoiser(
    ...     enc_embed_size=16,
    ...     dim=32,
    ...     depth=1,
    ...     n_blocks=1,
    ...     max_seq_len=8,
    ...     num_classes=4,
    ...     num_timesteps=50,
    ...     heads=2,
    ...     local_attn_window_size=8,
    ... )
    >>> diffusion = AptaDiffDiffusion(
    ...     denoise_fn=denoiser, num_classes=4, num_timesteps=50
    ... )
    >>> model_lightning = AptaDiffLightning(
    ...     diffusion, lr=1e-4, optimizer=torch.optim.AdamW
    ... )
    >>> config = model_lightning.configure_optimizers()
    >>> type(config["optimizer"]).__name__
    'AdamW'
    >>> config["lr_scheduler"]["interval"]
    'epoch'

    Training runs through a Lightning `Trainer`:

    >>> import lightning as L
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> tokens = torch.randint(0, 4, (4, 8))
    >>> x = torch.nn.functional.one_hot(tokens, num_classes=4)
    >>> x = x.permute(0, 2, 1).float()
    >>> z = torch.randn(4, 16)
    >>> loader = DataLoader(TensorDataset(x, z), batch_size=2)
    >>> trainer = L.Trainer(max_epochs=1)  # doctest: +SKIP
    >>> trainer.fit(model_lightning, loader)  # doctest: +SKIP
    """

    def __init__(
        self,
        diffusion: torch.nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0.0, 1.0], got: {gamma}.")

        self.diffusion = diffusion
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    def _step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        """Compute the bits per character loss for one batch and log it.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A pair `(x, z)`, where `x` is the one-hot encoded sequence batch of
            shape (batch_size, num_classes, seq_len) and `z` the latent
            conditioning vectors of shape (batch_size, enc_embed_size).
        stage : str
            Prefix of the logged metric name, `"train"` or `"val"`.

        Returns
        -------
        torch.Tensor
            Scalar loss for the batch, in bits per character.
        """
        x, z = batch
        num_tokens = x.size(0) * x.size(2)
        loss = -self.diffusion.log_prob(x, z).sum() / (math.log(2) * num_tokens)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Run one training step.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A pair `(x, z)`, where `x` is the one-hot encoded sequence batch of
            shape (batch_size, num_classes, seq_len) and `z` the latent
            conditioning vectors of shape (batch_size, enc_embed_size).
        batch_idx : int
            Index of the batch within the epoch.

        Returns
        -------
        torch.Tensor
            Scalar loss for the batch, in bits per character.
        """
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Run one validation step.

        The `Trainer` puts the model in eval mode before validating, so
        `log_prob` uses `eval_loss_type` and leaves the importance-sampling
        statistics `Lt_history` and `Lt_count` unchanged. When calling this
        method directly, for example in a test, call `.eval()` first.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            A pair `(x, z)` of one-hot sequences and latent conditioning
            vectors.
        batch_idx : int
            Index of the batch within the epoch.

        Returns
        -------
        torch.Tensor
            Scalar loss for the batch, in bits per character.
        """
        return self._step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        """Build the optimizer and its per-epoch learning-rate schedule.

        The optimizer is built for the current `lr` on every call, so a
        learning rate changed after construction takes effect.

        Returns
        -------
        dict
            Lightning optimizer configuration, holding the optimizer under
            `"optimizer"` and the `ExponentialLR` schedule under
            `"lr_scheduler"`. The schedule steps once per epoch.
        """
        optimizer = self.optimizer(
            self.parameters(), lr=self.lr, **(self.optimizer_kwargs or {})
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
