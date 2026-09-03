"""AptaDiff model architecture and diffusion training wrapper."""

__author__ = ["aditi-dsi"]
__all__ = ["Rezero", "AptaDiffDenoiser", "AptaDiffDiffusion"]

import warnings
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pyaptamer.aptadiff._functional import (
    compute_vlb_loss,
    cosine_alpha_schedule,
    log_one_minus_exp,
    multinomial_kl,
    q_forward,
    q_posterior,
)
from pyaptamer.aptadiff.layers import AptaDiffTransformerEmbedding


def _log_onehot_to_index(log_x: torch.Tensor) -> torch.Tensor:
    """Recover integer class indices from a log-one-hot tensor.

    Parameters
    ----------
    log_x : torch.Tensor
        Log-probabilities of shape (batch_size, num_classes, seq_len), as
        produced by :func:`_index_to_log_onehot` or the forward diffusion
        process.

    Returns
    -------
    torch.Tensor
        Integer class indices of shape (batch_size, seq_len), obtained by
        taking the argmax over the class dimension.
    """
    return log_x.argmax(dim=1)


def _log_sample_categorical(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Sample class indices from unnormalized logits via the Gumbel-max trick.

    Adds Gumbel noise to `logits` and takes the argmax, which is equivalent
    to sampling from the categorical distribution defined by
    ``softmax(logits, dim=1)`` without computing that softmax.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized log-probabilities of shape
        (batch_size, num_classes, seq_len).
    num_classes : int
        The number of unique nucleotides in the sequence.

    Returns
    -------
    torch.Tensor
        The sampled classes, re-encoded as a log-one-hot tensor of shape
        (batch_size, num_classes, seq_len).
    """
    uniform = torch.rand_like(logits)
    eps = torch.finfo(logits.dtype).tiny
    gumbel_noise = -torch.log(-torch.log(uniform + eps) + eps)
    sample = (gumbel_noise + logits).argmax(dim=1)

    return _index_to_log_onehot(sample, num_classes)


def _index_to_log_onehot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer class indices to a clamped log-one-hot tensor.

    Parameters
    ----------
    x : torch.Tensor
        Integer class indices of shape (batch_size, seq_len), with values
        in ``[0, num_classes)``.
    num_classes : int
        The number of unique nucleotides in the sequence.

    Returns
    -------
    torch.Tensor
        Log-one-hot encoding of shape (batch_size, num_classes, seq_len),
        computed in float32. Zero-probability classes are clamped to the
        smallest representable positive float32 value before taking the
        log, so the result is finite everywhere.

    Raises
    ------
    ValueError
        If `x` contains a class index greater than or equal to
        `num_classes`.
    """
    if torch.any((x < 0) | (x >= num_classes)):
        raise ValueError(
            f"x must contain class indices in [0, {num_classes}), got range "
            f"[{int(x.min())}, {int(x.max())}]."
        )

    x_onehot = F.one_hot(x, num_classes)
    x_onehot = x_onehot.permute(0, 2, 1).float()
    eps = torch.finfo(x_onehot.dtype).tiny

    return torch.log(x_onehot.clamp(min=eps))


class Rezero(nn.Module):
    """Learnable scalar gate, initialized to zero, that scales its input.

    `alpha` starts at zero, so the module's output is exactly zero for any
    input at initialization, and grows away from zero as `alpha` is
    updated during training.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the input by the learnable alpha parameter.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            `x` scaled elementwise by `alpha`, same shape as `x`.
        """
        return self.alpha * x


class AptaDiffDenoiser(nn.Module):
    """Denoising network for AptaDiff: a transformer backbone plus a Rezero gate.

    Predicts per-position nucleotide logits from a noisy sequence, its
    diffusion timestep, and a latent conditioning vector.

    Parameters
    ----------
    enc_embed_size : int
        Dimension of the input latent condition vector z.
    input_dim : int
        Size of the nucleotide vocabulary (number of unique input tokens).
    output_dim : int
        Dimension of the output sequence representations.
    dim : int
        Embedding and hidden dimension throughout the transformer blocks.
    depth : int
        Number of transformer layers per sequential block.
    n_blocks : int
        Number of outer sequential transformer blocks.
    max_seq_len : int
        Maximum sequence length of input aptamers.
    num_timesteps : int
        Total number of diffusion timesteps.
    heads : int, default=8
        Number of attention heads per transformer layer.
    attn_layer_dropout : float, default=0.0
        Dropout probability applied within attention blocks.
    n_local_attn_heads : int, default=0
        Number of heads dedicated to local windowed attention when using
        `transformer_type="linear"`. Ignored when using `"native"`.
    local_attn_window_size : int, default=128
        Window size used for axial positional indexing and local attention.
    transformer_type : {"native", "linear"}, default="native"
        The attention backend, passed through to
        `AptaDiffTransformerEmbedding`.

        - "native" : PyTorch's `nn.TransformerEncoderLayer`, giving exact
            attention and hardware acceleration. Recommended for typical
            aptamer lengths.
        - "linear" : the linear attention approximation used by the
            original AptaDiff paper. Use for strict reproducibility.

    """

    def __init__(
        self,
        enc_embed_size: int,
        input_dim: int,
        output_dim: int,
        dim: int,
        depth: int,
        n_blocks: int,
        max_seq_len: int,
        num_timesteps: int,
        heads: int = 8,
        attn_layer_dropout: float = 0.0,
        n_local_attn_heads: int = 0,
        local_attn_window_size: int = 128,
        transformer_type: Literal["native", "linear"] = "native",
    ):
        super().__init__()

        self.transformer = AptaDiffTransformerEmbedding(
            enc_embed_size=enc_embed_size,
            input_dim=input_dim,
            output_dim=output_dim,
            dim=dim,
            depth=depth,
            n_blocks=n_blocks,
            max_seq_len=max_seq_len,
            num_timesteps=num_timesteps,
            heads=heads,
            attn_layer_dropout=attn_layer_dropout,
            n_local_attn_heads=n_local_attn_heads,
            local_attn_window_size=local_attn_window_size,
            transformer_type=transformer_type,
        )
        self.rezero = Rezero()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Predict per-class denoising logits for a noisy sequence.

        Parameters
        ----------
        x : torch.Tensor
            Sequence token index tensor of shape (batch_size, seq_len).
        t : torch.Tensor
            Diffusion timesteps tensor of shape (batch_size,).
        z : torch.Tensor
            Latent conditioning vector of shape (batch_size, enc_embed_size).

        Returns
        -------
        torch.Tensor
            Per-class logits of shape (batch_size, num_classes, seq_len).
        """
        out = self.transformer(x, t, z)
        out = out.permute(0, 2, 1)  # (batch, num_classes, seq_len)
        out = self.rezero(out)
        return out


class AptaDiffDiffusion(nn.Module):
    """Multinomial diffusion wrapper computing the training loss for a denoiser.

    Owns the noise schedule and the importance-sampling statistics, and exposes
    the training-loss path (`log_prob`) built around the VLB loss in
    `pyaptamer.aptadiff._functional`. Does not implement sampling/generation.

    Parameters
    ----------
    denoise_fn : nn.Module
        The denoising network, e.g. an `AptaDiffDenoiser` instance. Must
        accept `(x, t, z)` and return logits of shape
        `(batch_size, num_classes, seq_len)`.
    num_classes : int, default=4
        The number of unique nucleotides in the sequence.
    num_timesteps : int, default=1000
        Total number of diffusion timesteps.
    loss_type : {"vb_stochastic", "vb_all"}, default="vb_stochastic"
        Which variational bound to optimize.
        - "vb_stochastic" : one importance-sampled timestep per training
          step. This is the default.
        - "vb_all" : the exact bound, summed over every timestep. Costs
          `num_timesteps` denoiser forward passes per step.

    parametrization : {"x0", "direct"}, default="x0"
        - "x0" : the denoiser predicts the clean sequence x0. That
          prediction is then turned into a reverse-step distribution by
          `q_posterior`.
        - "direct" : the denoiser predicts the reverse-step distribution
          itself, and its output is used unchanged.

    Attributes
    ----------
    log_alpha : torch.Tensor
        Buffer of shape (num_timesteps,) holding log alpha_t, the per-step
        log probability that a token keeps its current class.
    log_1m_alpha : torch.Tensor
        Buffer of shape (num_timesteps,) holding log(1 - alpha_t).
    log_alphabar : torch.Tensor
        Buffer of shape (num_timesteps,) holding log alphabar_t, the
        cumulative sum of `log_alpha` through step t.
    log_1m_alphabar : torch.Tensor
        Buffer of shape (num_timesteps,) holding log(1 - alphabar_t).
    Lt_history : torch.Tensor
        Buffer of shape (num_timesteps,) holding an exponential moving
        average of the squared VLB term at each timestep. Used to build the
        importance-sampling proposal distribution.
    Lt_count : torch.Tensor
        Buffer of shape (num_timesteps,) counting how many times each
        timestep has been sampled. Importance sampling stays disabled until
        every timestep has more than 10 recorded losses.

    Raises
    ------
    ValueError
        If `loss_type` is not one of `"vb_stochastic"` or `"vb_all"`, or if
        `parametrization` is not one of `"x0"` or `"direct"`.
    """

    def __init__(
        self,
        denoise_fn: nn.Module,
        num_classes: int = 4,
        num_timesteps: int = 1000,
        loss_type: Literal["vb_stochastic", "vb_all"] = "vb_stochastic",
        parametrization: Literal["x0", "direct"] = "x0",
    ):
        super().__init__()

        if loss_type not in ("vb_stochastic", "vb_all"):
            raise ValueError(
                f"loss_type must be 'vb_stochastic' or 'vb_all', got: {loss_type}"
            )
        if parametrization not in ("x0", "direct"):
            raise ValueError(
                f"parametrization must be 'x0' or 'direct', got: {parametrization}"
            )

        if loss_type == "vb_all":
            warnings.warn(
                "loss_type='vb_all' evaluates the bound on every timestep, so "
                f"each step runs {num_timesteps} denoiser forward passes. Use "
                "'vb_stochastic' unless you specifically need the exact bound.",
                stacklevel=2,
            )

        self.num_classes = num_classes
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization

        alphas = cosine_alpha_schedule(num_timesteps)
        log_alpha = torch.log(alphas)
        log_alphabar = torch.cumsum(log_alpha, dim=0)

        log_1m_alpha = log_one_minus_exp(log_alpha)
        log_1m_alphabar = log_one_minus_exp(log_alphabar)

        self.register_buffer("log_alpha", log_alpha)
        self.register_buffer("log_1m_alpha", log_1m_alpha)
        self.register_buffer("log_alphabar", log_alphabar)
        self.register_buffer("log_1m_alphabar", log_1m_alphabar)

        self.register_buffer("Lt_history", torch.zeros(num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(num_timesteps))

    def predict_start(
        self, log_xt: torch.Tensor, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Run the denoiser and return log-softmax predictions of x0.

        Parameters
        ----------
        log_xt : torch.Tensor
            Log-one-hot noisy sequence at step t, shape
            (batch_size, num_classes, seq_len).
        t : torch.Tensor
            Diffusion timesteps tensor of shape (batch_size,).
        z : torch.Tensor
            Latent conditioning vector of shape (batch_size, enc_embed_size).

        Returns
        -------
        torch.Tensor
            Log-probabilities over the reconstructed clean sequence, shape
            (batch_size, num_classes, seq_len).

        Raises
        ------
        ValueError
            If `denoise_fn` returns logits whose shape is not
            (batch_size, num_classes, seq_len).
        """
        xt = _log_onehot_to_index(log_xt)
        out = self.denoise_fn(xt, t, z)

        expected_shape = (xt.size(0), self.num_classes, *xt.shape[1:])
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"denoise_fn must return logits of shape {expected_shape}, got "
                f"{tuple(out.shape)}."
            )

        log_pred = F.log_softmax(out, dim=1)

        return log_pred

    def predict_reverse_step(
        self, log_x: torch.Tensor, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Predict the reverse-step distribution, per `self.parametrization`.

        Parameters
        ----------
        log_x : torch.Tensor
            Log-one-hot noisy sequence at step t, shape
            (batch_size, num_classes, seq_len).
        t : torch.Tensor
            Diffusion timesteps tensor of shape (batch_size,).
        z : torch.Tensor
            Latent conditioning vector of shape (batch_size, enc_embed_size).

        Returns
        -------
        torch.Tensor
            Log-probabilities for the model's reverse-step prediction,
            shape (batch_size, num_classes, seq_len).
            If `self.parametrization` is `"x0"`, this is the true posterior
            q(x_{t-1} | x_t, predicted x0)
            If `"direct"`, this is the denoiser's raw prediction.
        """
        if torch.any((t < 0) | (t >= self.num_timesteps)):
            raise ValueError(
                f"t must contain timesteps in [0, {self.num_timesteps}), got range "
                f"[{int(t.min())}, {int(t.max())}]."
            )

        if self.parametrization == "x0":
            log_x_recon = self.predict_start(log_x, t, z)
            log_model_pred = q_posterior(
                log_x0=log_x_recon,
                log_xt=log_x,
                t=t,
                log_alpha=self.log_alpha,
                log_alphabar=self.log_alphabar,
            )
        else:
            log_model_pred = self.predict_start(log_x, t, z)

        return log_model_pred

    def q_sample(self, log_x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample a noisy sequence at timestep t from the clean data.

        Parameters
        ----------
        log_x0 : torch.Tensor
            Log-one-hot clean sequence, shape
            (batch_size, num_classes, seq_len).
        t : torch.Tensor
            Diffusion timesteps tensor of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Log-one-hot noisy sequence at step t, shape
            (batch_size, num_classes, seq_len).
        """
        if torch.any((t < 0) | (t >= self.num_timesteps)):
            raise ValueError(
                f"t must contain timesteps in [0, {self.num_timesteps}), got range "
                f"[{int(t.min())}, {int(t.max())}]."
            )

        log_xt_probs = q_forward(log_x0, t, self.log_alphabar)
        log_sample = _log_sample_categorical(log_xt_probs, self.num_classes)

        return log_sample

    def kl_prior(self, log_x0: torch.Tensor) -> torch.Tensor:
        """Compute the KL divergence between the noise prior and a uniform prior.

        Evaluates q(x_t | x0) at the final timestep (t = num_timesteps - 1)
        and compares it against a uniform categorical distribution over
        num_classes, which is what the forward process is designed to
        converge to.

        Parameters
        ----------
        log_x0 : torch.Tensor
            Log-one-hot clean sequence, shape
            (batch_size, num_classes, seq_len).

        Returns
        -------
        torch.Tensor
            Per-sequence KL divergence, shape (batch_size,), summed over
            sequence positions.
        """
        ones = torch.ones(log_x0.size(0), device=log_x0.device, dtype=torch.long)

        log_final_noise_probs = q_forward(
            log_x0=log_x0,
            t=(self.num_timesteps - 1) * ones,
            log_alphabar=self.log_alphabar,
        )

        log_uniform_prior = -torch.log(
            self.num_classes * torch.ones_like(log_final_noise_probs)
        )
        kl_div = multinomial_kl(log_final_noise_probs, log_uniform_prior)

        return torch.sum(kl_div, dim=1)

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        method: Literal["uniform", "importance"] = "uniform",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of timesteps and their associated sampling probabilities.

        Parameters
        ----------
        batch_size : int
            Number of timesteps to sample.
        device : torch.device
            Device the returned tensors are placed on.
        method : {"unform", "importance"}, default="uniform"
            - `"uniform"`: sample timesteps uniformly at random
            - `"importance"`: sample proportionally to the square root of
            each timestep's exponentially-averaged squared loss,
            `Lt_history`, falling back to `"uniform"` until at least 10
            losses have been recorded for every timestep.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            `sampled_timesteps` of shape (batch_size,), dtype `torch.long`,
            and `sampled_probs` of shape (batch_size,): the probability
            under `method` of having drawn each sampled timestep.

        Raises
        ------
        ValueError
            If `method` is not one of `"uniform"` or `"importance"`.
        """
        if method not in ("uniform", "importance"):
            raise ValueError(f"Unknown sample time method: {method}")

        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(batch_size, device, method="uniform")

            sampling_scores = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            sampling_scores[0] = sampling_scores[
                1
            ]  # match t=0 score to t=1 to prevent scale distortion
            all_probs = sampling_scores / sampling_scores.sum()
            sampled_timesteps = torch.multinomial(
                all_probs, num_samples=batch_size, replacement=True
            )
            sampled_probs = all_probs.gather(dim=0, index=sampled_timesteps)

            return sampled_timesteps.to(device), sampled_probs.to(device)

        else:
            sampled_timesteps = torch.randint(
                0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long
            )
            sampled_probs = (
                torch.ones_like(sampled_timesteps).float() / self.num_timesteps
            )

            return sampled_timesteps, sampled_probs

    def compute_full_vlb(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute the full variational lower bound, summed over every timestep.

        Used by the `"vb_all"` loss type. Unlike `_train_loss`'s
        `"vb_stochastic"` path, this evaluates every timestep from 0 to
        `self.num_timesteps - 1` rather than a single sampled one, so its
        cost scales linearly with `self.num_timesteps`.

        Parameters
        ----------
        x : torch.Tensor
            Integer-encoded clean sequence, shape (batch_size, seq_len).
            Converted internally to log-one-hot.
        z : torch.Tensor
            Latent conditioning vector of shape (batch_size, enc_embed_size).

        Returns
        -------
        torch.Tensor
            Per-sequence full VLB, shape (batch_size,).
        """
        log_x0 = _index_to_log_onehot(x, self.num_classes)
        total_loss = 0
        for t_idx in range(0, self.num_timesteps):
            t_array = torch.full(
                (log_x0.size(0),), t_idx, device=log_x0.device, dtype=torch.long
            )
            log_xt = self.q_sample(log_x0, t_array)

            if torch.is_grad_enabled():
                log_pred = checkpoint(
                    self.predict_reverse_step,
                    log_xt,
                    t_array,
                    z,
                    use_reentrant=False,
                )
            else:
                log_pred = self.predict_reverse_step(log_xt, t_array, z)

            loss = compute_vlb_loss(
                log_x0=log_x0,
                log_xt=log_xt,
                t=t_array,
                log_alpha=self.log_alpha,
                log_alphabar=self.log_alphabar,
                log_pred=log_pred,
            )
            total_loss = total_loss + loss

        total_loss = total_loss + self.kl_prior(log_x0)
        return total_loss

    def _train_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute the training loss for one batch, per `self.loss_type`.

        Parameters
        ----------
        x : torch.Tensor
            Integer-encoded clean sequence, shape (batch_size, seq_len).
        z : torch.Tensor
            Latent conditioning vector of shape (batch_size, enc_embed_size).

        Returns
        -------
        torch.Tensor
            Per-sequence negative VLB (the training loss), shape
            (batch_size,).
        """
        if self.loss_type == "vb_stochastic":
            sampled_timesteps, sampled_probs = self.sample_time(
                x.size(0), x.device, "importance"
            )
            log_x0 = _index_to_log_onehot(x, self.num_classes)
            log_xt = self.q_sample(log_x0, sampled_timesteps)
            log_pred = self.predict_reverse_step(log_xt, sampled_timesteps, z)

            loss = compute_vlb_loss(
                log_x0=log_x0,
                log_xt=log_xt,
                t=sampled_timesteps,
                log_alpha=self.log_alpha,
                log_alphabar=self.log_alphabar,
                log_pred=log_pred,
            )

            Lt_sqrd = loss.pow(2)
            Lt_sqrd_prev = self.Lt_history.gather(dim=0, index=sampled_timesteps)
            new_Lt_history = (0.1 * Lt_sqrd + 0.9 * Lt_sqrd_prev).detach()

            self.Lt_history.scatter_(dim=0, index=sampled_timesteps, src=new_Lt_history)
            self.Lt_count.scatter_add_(
                dim=0, index=sampled_timesteps, src=torch.ones_like(Lt_sqrd)
            )

            kl_prior = self.kl_prior(log_x0)
            total_loss = (loss / sampled_probs) + kl_prior

            return -total_loss

        else:
            return -self.compute_full_vlb(x, z)

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns a per-sequence lower bound on the log-probability of `x`.

        This is the negated variational bound, i.e. an ELBO - higher is
        better.

        Dispatches to `_train_loss` in train mode. In eval mode, computes
        the same single-sampled-timestep estimate but doesn't update the
        `Lt_history`/`Lt_count` importance-sampling statistics.

        Parameters
        ----------
        x : torch.Tensor
            Integer-encoded sequence, shape (batch_size, seq_len).
        z : torch.Tensor
            Latent conditioning vector of shape (batch_size, enc_embed_size).

        Returns
        -------
        torch.Tensor
            Per-sequence log-probability lower bound, shape (batch_size,).

        Raises
        ------
        ValueError
            If `x` and `z` have different batch sizes.
        """
        if x.size(0) != z.size(0):
            raise ValueError(
                f"x and z must have the same batch size, got {x.size(0)} and "
                f"{z.size(0)}."
            )

        if self.training:
            return self._train_loss(x, z)

        else:
            log_x0 = _index_to_log_onehot(x, self.num_classes)
            sampled_timesteps, sampled_probs = self.sample_time(
                x.size(0), x.device, "importance"
            )

            log_xt = self.q_sample(log_x0, sampled_timesteps)
            log_pred = self.predict_reverse_step(log_xt, sampled_timesteps, z)

            loss = compute_vlb_loss(
                log_x0=log_x0,
                log_xt=log_xt,
                t=sampled_timesteps,
                log_alpha=self.log_alpha,
                log_alphabar=self.log_alphabar,
                log_pred=log_pred,
            )

            kl_prior = self.kl_prior(log_x0)
            total_loss = (loss / sampled_probs) + kl_prior

            return -total_loss
