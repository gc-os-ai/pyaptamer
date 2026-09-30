"""Scikit-learn estimator that trains AptaDiff and generates sequences."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiffGenerator"]

import tempfile
from typing import Any, Literal

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data
from torch.utils.data import DataLoader, TensorDataset

from pyaptamer.aptadiff._model import (
    AptaDiffDenoiser,
    AptaDiffDiffusion,
    _log_sample_categorical,
)
from pyaptamer.aptadiff._train_lightning import AptaDiffLightning


class AptaDiffGenerator(BaseEstimator):
    """Train AptaDiff's diffusion model and generate aptamer sequences.

    Learns to generate sequences conditioned on latent vectors, such as the latent
    representations of a RaptGen VAE. `fit` trains on pairs of latent vectors `X`
    and one-hot encoded sequences `y`. `predict` generates one sequence for one
    row of `X` by running the reverse diffusion process from uniform noise.

    Sequences use the flat one-hot layout of `SequenceOneHotEncoder` where a sequence
    of length ``L`` over ``num_classes`` classes has ``num_classes * L``
    columns.

    Training runs on the fastest available device (CUDA, MPS or CPU), using a
    single device. A share of the data, set by `validation_fraction`, is held
    out, and the best weights (the weights with the lowest validation loss)
    from the epoch are kept.

    Parameters
    ----------
    num_classes : int, optional, default=4
        The number of unique nucleotides in the sequence.
    num_timesteps : int, optional, default=1000
        Total number of diffusion timesteps.
        `predict` runs one denoiser pass per timestep.
    dim : int, optional, default=512
        Embedding and hidden dimension throughout the transformer blocks.
        This must be an even number.
    depth : int, optional, default=12
        Number of transformer layers per sequential block.
    n_blocks : int, optional, default=1
        Number of outer sequential transformer blocks.
    heads : int, optional, default=16
        Number of attention heads per transformer layer.
        `dim` must be divisible by `heads`.
    attn_layer_dropout : float, optional, default=0.0
        Dropout probability applied within attention blocks.
    n_local_attn_heads : int, optional, default=0
        Number of heads dedicated to local windowed attention when using
        `transformer_type="linear"`. Ignored when using `"native"`.
    local_attn_window_size : int, optional, default=1
        Window size used for axial positional indexing and local attention.
        The sequence length of `y` must be divisible by it.
    transformer_type : {"native", "linear"}, optional, default="native"
        The attention backend, passed to :class:`~pyaptamer.aptadiff.AptaDiffDenoiser`.

        - "native" : PyTorch's `nn.TransformerEncoderLayer`, giving exact
            attention and hardware acceleration. Recommended for typical
            aptamer lengths.
        - "linear" : the linear attention approximation used by the
            original AptaDiff paper. Use this for strict reproducibility.
    loss_type : {"vb_stochastic", "vb_all"}, optional, default="vb_stochastic"
        Chooses which variational bound to optimize in training mode.
        - "vb_stochastic" : one importance-sampled timestep per training
          step. This is the default.
        - "vb_all" : Computes the loss across all timesteps instead of
          randomly sampling just one. This provides a more stable estimate,
          but costs `num_timesteps` denoiser forward passes per training step.
    eval_loss_type : {"vb_stochastic", "vb_all"}, optional, default="vb_stochastic"
        Chooses which variational bound to compute in eval mode.
        - "vb_stochastic" : one importance-sampled timestep per call. This is
          the default. Original implementation strictly uses this always for eval mode.
          So, choose it for strict reproducibility.
        - "vb_all" : Computes the loss across all timesteps instead of
          randomly sampling just one. This provides a more stable validation
          metric, but costs `num_timesteps` denoiser forward passes per eval call.

    parametrization : {"x0", "direct"}, optional, default="x0"
        - "x0" : the denoiser predicts the clean sequence x0. That
          prediction is then turned into a reverse-step distribution by
          `q_posterior`.
        - "direct" : the denoiser predicts the reverse-step distribution
          itself, and its output is used unchanged.
    max_epochs : int, optional, default=1000
        Number of training epochs. This must be at least 1. There is no early
        stopping. The weights from the epoch with the lowest validation loss
        are kept.
    batch_size : int, optional, default=32
        Training batch size, as in the original implementation's training script
        (the paper states 64). `predict` generates in batches of the same size.
    validation_fraction : float, optional, default=0.1
        Share of the data held out to compute the validation loss that selects
        the best epoch. Must lie in (0.0, 1.0). The default matches the original
        9:1 split.
    lr : float, optional, default=1e-4
        Initial learning rate, as used by the original implementation.
        Overrides PyTorch's default learning rate for the chosen optimizer.
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
    gamma : float, optional, default=0.99
        Per-epoch decay factor of the `ExponentialLR` schedule. Must lie
        in (0.0, 1.0]. Value 1.0 keeps the learning rate constant. Because the decay
        is exponential, values below 1.0 shrink the learning rate quickly over long
        runs, so later epochs update the model very little. Raise gamma towards 1.0
        to train longer at a useful rate.
        After `n` epochs the learning rate is ``lr * gamma**n``.
    random_state : int, optional, default=None
        Seed for the train/validation split, weight initialisation, batch
        shuffling, the diffusion noise during training, and sampling in
        `predict`. The global NumPy and Torch random states are left unchanged.
    verbose : int, optional, default=0
        If non-zero, show Lightning's progress bar and model summary during
        `fit`.

    Attributes
    ----------
    diffusion_ : AptaDiffDiffusion
        The trained diffusion model, in eval mode, on the device it was trained
        on. `predict` runs on that device. An estimator pickle taken while `diffusion_`
        is on a CUDA or MPS device only loads where that device type is available.
        Call ``diffusion_.cpu()`` before saving to get a file that loads on any machine.
    seq_len_ : int
        Length of the training sequences, and of the sequences generated by `predict`.
    n_features_in_ : int
        Dimension of the latent vectors in `X`.

    References
    ----------
    .. [1] Wang, Z., et al. "AptaDiff: de novo design and optimization of
           aptamers based on diffusion models." Briefings in Bioinformatics,
           25(6), bbae517 (2024). https://doi.org/10.1093/bib/bbae517

           Original implementation: https://github.com/wz-create/AptaDiff

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptadiff import AptaDiffGenerator
    >>> rng = np.random.default_rng(0)
    >>> tokens = rng.integers(0, 4, size=(20, 8))
    >>> y = np.eye(4, dtype=np.float32)[tokens].transpose(0, 2, 1).reshape(20, -1)
    >>> X = rng.normal(size=(20, 2)).astype(np.float32)
    >>> generator = AptaDiffGenerator(
    ...     num_timesteps=5, dim=32, depth=1, max_epochs=1, random_state=0
    ... )
    >>> generator.fit(X, y)  # doctest: +ELLIPSIS
    AptaDiffGenerator(...)
    >>> generator.predict(X[:3]).shape
    (3, 32)
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_timesteps: int = 1000,
        dim: int = 512,
        depth: int = 12,
        n_blocks: int = 1,
        heads: int = 16,
        attn_layer_dropout: float = 0.0,
        n_local_attn_heads: int = 0,
        local_attn_window_size: int = 1,
        transformer_type: Literal["native", "linear"] = "native",
        loss_type: Literal["vb_stochastic", "vb_all"] = "vb_stochastic",
        eval_loss_type: Literal["vb_stochastic", "vb_all"] = "vb_stochastic",
        parametrization: Literal["x0", "direct"] = "x0",
        max_epochs: int = 1000,
        batch_size: int = 32,
        validation_fraction: float = 0.1,
        lr: float = 1e-4,
        optimizer_name: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        gamma: float = 0.99,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.dim = dim
        self.depth = depth
        self.n_blocks = n_blocks
        self.heads = heads
        self.attn_layer_dropout = attn_layer_dropout
        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn_window_size = local_attn_window_size
        self.transformer_type = transformer_type
        self.loss_type = loss_type
        self.eval_loss_type = eval_loss_type
        self.parametrization = parametrization
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.gamma = gamma
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AptaDiffGenerator":
        """Train the diffusion model.

        Parameters
        ----------
        X : array-like
            Latent conditioning vectors, one per sequence, shape
            (n_samples, n_latent).
        y : array-like
            One-hot encoded sequences in the flat layout described above, shape
            (n_samples, num_classes * seq_len).

        Returns
        -------
        AptaDiffGenerator
            The fitted estimator.

        Raises
        ------
        ValueError
            - If the number of columns of `y` is not a multiple of `num_classes`,
            - If y is not properly one-hot encoded (e.g., if raw sequences are
            passed instead of a flattened array of 1s and 0s).
            - If the optimizer, loss or `gamma` settings are invalid during
            model construction.
        """
        X, y = validate_data(self, X, y, multi_output=True, dtype=np.float32)
        y = y.astype(np.float32, copy=False)

        seq_len, remainder = divmod(y.shape[1], self.num_classes)
        if remainder:
            raise ValueError(
                f"y has {y.shape[1]} columns, which is not a multiple of "
                f"num_classes={self.num_classes}."
            )

        y = y.reshape(len(y), self.num_classes, seq_len)
        if not (np.isin(y, (0, 1)).all() and (y.sum(axis=1) == 1).all()):
            raise ValueError(
                "y must hold one-hot encoded sequences, as produced by "
                "SequenceOneHotEncoder, with exactly one 1 per positiaon across "
                "the num_classes classes. Raw token indices are not accepted."
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, random_state=self.random_state
        )

        with torch.random.fork_rng(enabled=self.random_state is not None):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)

            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(y_train), torch.from_numpy(X_train)),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(y_val), torch.from_numpy(X_val)),
                batch_size=self.batch_size,
            )
            model = self._build_model(seq_len)

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                checkpoint = ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    save_weights_only=True,
                )
                trainer = L.Trainer(
                    max_epochs=self.max_epochs,
                    accelerator="auto",
                    devices=1,
                    logger=False,
                    callbacks=[checkpoint],
                    enable_progress_bar=bool(self.verbose),
                    enable_model_summary=bool(self.verbose),
                )
                trainer.fit(model, train_loader, val_loader)

                best = torch.load(checkpoint.best_model_path, map_location="cpu")
                model.load_state_dict(best["state_dict"])

        self.diffusion_ = model.diffusion.to(trainer.strategy.root_device).eval()
        self.seq_len_ = seq_len
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Generate one sequence per latent vector.

        Runs the reverse diffusion chain from uniform noise, conditioned on
        each row of `X`, on the device holding `diffusion_`.

        Parameters
        ----------
        X : array-like
            Latent conditioning vectors, shape (n_samples, n_latent).

        Returns
        -------
        numpy.ndarray
            Generated sequences as float32, one-hot encoded in the same flat
            layout as `y` in `fit`, shape (n_samples, num_classes * seq_len_).
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float32)
        num_classes = self.diffusion_.num_classes
        device = next(self.diffusion_.parameters()).device

        with torch.random.fork_rng(enabled=self.random_state is not None):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)

            with torch.inference_mode():
                latents = torch.from_numpy(X).to(device)
                tokens = torch.cat(
                    [self._sample(z) for z in torch.split(latents, self.batch_size)]
                )
                onehot = F.one_hot(tokens, num_classes).permute(0, 2, 1)
                return onehot.reshape(len(X), -1).float().cpu().numpy()

    def _build_model(self, seq_len: int) -> AptaDiffLightning:
        """Build the untrained model and wrap it for Lightning training.

        Parameters
        ----------
        seq_len : int
            Length of the training sequences, used as the denoiser's
            `max_seq_len`.

        Returns
        -------
        AptaDiffLightning
            The denoiser and diffusion model, configured from this estimator's
            parameters.
        """
        denoiser = AptaDiffDenoiser(
            enc_embed_size=self.n_features_in_,
            dim=self.dim,
            depth=self.depth,
            n_blocks=self.n_blocks,
            max_seq_len=seq_len,
            num_classes=self.num_classes,
            num_timesteps=self.num_timesteps,
            heads=self.heads,
            attn_layer_dropout=self.attn_layer_dropout,
            n_local_attn_heads=self.n_local_attn_heads,
            local_attn_window_size=self.local_attn_window_size,
            transformer_type=self.transformer_type,
        )
        diffusion = AptaDiffDiffusion(
            denoise_fn=denoiser,
            num_classes=self.num_classes,
            num_timesteps=self.num_timesteps,
            loss_type=self.loss_type,
            eval_loss_type=self.eval_loss_type,
            parametrization=self.parametrization,
        )
        model_lightning = AptaDiffLightning(
            diffusion=diffusion,
            optimizer_name=self.optimizer_name,
            optimizer_kwargs=self.optimizer_kwargs,
            lr=self.lr,
            gamma=self.gamma,
        )

        return model_lightning

    def _sample(self, z: torch.Tensor) -> torch.Tensor:
        """Run the reverse diffusion chain for one batch of latent vectors.

        Starts from uniform noise over the classes and denoises from the last
        timestep to the first, conditioned on `z`.

        Parameters
        ----------
        z : torch.Tensor
            Latent conditioning vector of shape (batch_size, enc_embed_size).
            It must be on the same device as `diffusion_`.

        Returns
        -------
        torch.Tensor
            Sampled class indices, shape (batch_size, seq_len_).
        """
        num_classes = self.diffusion_.num_classes
        batch_size = len(z)

        log_x = _log_sample_categorical(
            torch.zeros(batch_size, num_classes, self.seq_len_, device=z.device),
            num_classes,
        )
        for step in reversed(range(self.diffusion_.num_timesteps)):
            t = torch.full((batch_size,), step, dtype=torch.long, device=z.device)
            log_pred = self.diffusion_.predict_reverse_step(log_x, t, z)
            log_x = _log_sample_categorical(log_pred, num_classes)

        return log_x.argmax(dim=1)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.target_tags.required = True
        return tags
