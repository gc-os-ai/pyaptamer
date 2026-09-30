"""LightningModule defining the training steps for AptaDiff."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiffLightning"]

import inspect
import math
from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import Tensor


def _build_optimizer_registry() -> dict[str, type[torch.optim.Optimizer]]:
    """Collect the optimizer classes exposed by `torch.optim`.

    Returns
    -------
    dict[str, type[torch.optim.Optimizer]]
        Mapping from each optimizer's class name, e.g. `"AdamW"`, to the class object.
    """
    registry = {}
    for name in dir(torch.optim):
        if name.startswith("_"):
            continue
        obj = getattr(torch.optim, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, torch.optim.Optimizer)
            and obj is not torch.optim.Optimizer
        ):
            registry[name] = obj
    return registry


_OPTIMIZER_CLASSES = _build_optimizer_registry()


def _resolve_optimizer_cls(optimizer_name: str) -> type[torch.optim.Optimizer]:
    """Look up an optimizer class in `torch.optim` by name, case-insensitively.

    Parameters
    ----------
    optimizer_name : str
        Name of an optimizer class in `torch.optim`. This is case-insensitive.

    Returns
    -------
    type[torch.optim.Optimizer]
        The matching optimizer class.

    Raises
    ------
    ValueError
        If `optimizer_name` matches no optimizer, after ignoring the case.
    """
    for name, optimizer_cls in _OPTIMIZER_CLASSES.items():
        if name.lower() == optimizer_name.lower():
            return optimizer_cls

    raise ValueError(
        "optimizer_name must name an optimizer in torch.optim, in any case, got "
        f"{optimizer_name!r}. Available optimizers: "
        f"{list(_OPTIMIZER_CLASSES)}."
    )


def _check_optimizer_kwargs(
    optimizer_cls: type[torch.optim.Optimizer], optimizer_kwargs: Mapping[str, Any]
) -> None:
    """Check that `optimizer_kwargs` contains only args that `optimizer_cls` accepts.

    Only the argument names are checked. Their values are left to the optimizer,
    which validates them when it is built.

    Parameters
    ----------
    optimizer_cls : type[torch.optim.Optimizer]
        The optimizer the arguments are meant for.
    optimizer_kwargs : Mapping[str, Any]
        Arguments for the optimizer other than the model weights (params) and the
        learning rate (lr).

    Raises
    ------
    ValueError
        If `optimizer_kwargs` contains `lr` or `params`, or any other key that
        `optimizer_cls` does not accept.
    """
    if "lr" in optimizer_kwargs or "params" in optimizer_kwargs:
        raise ValueError(
            "optimizer_kwargs must not contain 'lr' or 'params'. The model weights "
            "are handled automatically by PyTorch Lightning, and the learning rate "
            "is supplied by the `lr` argument."
        )

    signature = inspect.signature(optimizer_cls).parameters

    accepted = [name for name in signature if name not in ("lr", "params")]
    unknown = [key for key in optimizer_kwargs if key not in accepted]
    if unknown:
        raise ValueError(
            f"{optimizer_cls.__name__} does not accept {unknown}. "
            f"Accepted arguments: {accepted}."
        )


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
    optimizer_name : str, optional, default="adam"
        Name of an optimizer class in `torch.optim`, in any case, e.g.
        `"adam"`, `"AdamW"` or `"SGD"`. The default is `adam`, the optimizer
        the original implementation trained with. This supports any optimizer available
        in the current installed torch version that is compatible with dense gradients
        and parameters of any number of dimensions.
    optimizer_kwargs : dict, optional, default=None
        Keyword arguments to be passed to the optimizer, for example
        ``{"momentum": 0.9}`` for `"sgd"` or ``{"weight_decay": 0.01}`` for
        `"adamw"`. Keys are checked against the optimizer's signature. If an optimizer
        does not accept a particular key, it raises `ValueError` at construction.
        This must not contain `lr` or `params`. PyTorch's defaults will be applied to
        anything omitted. With the default `"adam"`, `optimizer_kwargs=None`
        reproduces the original configuration exactly.
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
        - If `optimizer_name` doesn't matches any optimizer in `torch.optim`.
        - If `optimizer_kwargs` contains `lr` or `params`, or a key the chosen
          optimizer does not accept.
        - If `gamma` lies outside (0.0, 1.0].

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
    >>> model_lightning = AptaDiffLightning(diffusion, lr=1e-4, optimizer_name="adamw")
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
        optimizer_name: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self.get_optimizer_cls_and_kwargs(optimizer_name, optimizer_kwargs, lr)

        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0.0, 1.0], got: {gamma}.")

        self.diffusion = diffusion
        self.lr = lr
        self.gamma = gamma
        self.optimizer_name = optimizer_name
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

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        optimizer_name: str,
        optimizer_kwargs: Mapping[str, Any] | None,
        lr: float,
    ) -> tuple[type[torch.optim.Optimizer], dict[str, Any]]:
        """Resolve an optimizer name and arrange the arguments to build it.

        Looks up for `optimizer_name` among the optimizers in `torch.optim`,
        ignoring the case, and checks `optimizer_kwargs` against that optimizer's
        signature. This can check a configuration independently before any model
        or data exists.

        Parameters
        ----------
        optimizer_name : str
            Name of an optimizer class in `torch.optim`, in any case, e.g.
            `"adam"`, `"AdamW"` or `"SGD"`.
        optimizer_kwargs : Mapping[str, Any] or None
            Arguments for the optimizer other than the model weights (params) and the
            learning rate (lr).
        lr : float
            Learning rate.

        Returns
        -------
        optimizer_cls : type[torch.optim.Optimizer]
            The resolved optimizer class.
        optimizer_kwargs : dict[str, Any]
            A new dict holding `lr` followed by the entries of
            `optimizer_kwargs`.

        Raises
        ------
        ValueError
            If `optimizer_name` doesn't matches any optimizer in `torch.optim`,
            or if `optimizer_kwargs` contains `lr`, `params`, or a key the optimizer
            does not accept.
        """
        optimizer_cls = _resolve_optimizer_cls(optimizer_name)
        if optimizer_kwargs:
            _check_optimizer_kwargs(optimizer_cls, optimizer_kwargs)
            return optimizer_cls, {"lr": lr, **optimizer_kwargs}
        return optimizer_cls, {"lr": lr}

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
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.optimizer_name, self.optimizer_kwargs, self.lr
        )
        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
