__author__ = ["aditi-dsi"]
__all__ = ["Rezero", "AptaDiffDenoiser", "AptaDiffDiffusion"]


import torch
import torch.nn as nn
import torch.nn.functional as F

from pyaptamer.aptadiff._functional import (
    compute_vlb_loss,
    cosine_alpha_schedule,
    log_one_minus_exp,
    q_forward,
    q_posterior,
)
from pyaptamer.aptadiff.layers._transformer import AptaDiffTransformerEmbedding


def _log_onehot_to_index():
    pass


def _log_sample_categorical():
    pass


def _index_to_log_onehot():
    pass


class Rezero(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x


class AptaDiffDenoiser(nn.Module):
    def __init__(
        self,
        enc_embed_size,
        input_dim,
        output_dim,
        dim,
        depth,
        n_blocks,
        max_seq_len,
        num_timesteps,
        heads=8,
        attn_layer_dropout=0.0,
        n_local_attn_heads=0,
        local_attn_window_size=128,
        transformer_type="native",
        **kwargs,
    ):
        super().__init__()

        self.transformer = AptaDiffTransformerEmbedding(
            self,
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

    def forward(self, x, t, z):
        out = self.transformer(x, t, z)  # (batch, seq_len, num_classes)
        out = out.permute(0, 2, 1)  # (batch, num_classes, seq_len)
        out = self.rezero(out)
        return out


class AptaDiffDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        num_classes: int = 4,
        timesteps: int = 1000,
        loss_type: str = "vb_stochastic",
        parametrization: str = "x0",
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

        self.num_classes = num_classes
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        alphas = cosine_alpha_schedule(timesteps)
        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)

        log_1m_alpha = log_one_minus_exp(log_alpha)
        log_1m_cumprod_alpha = log_one_minus_exp(log_cumprod_alpha)

        self.register_buffer("log_alpha", log_alpha)
        self.register_buffer("log_1m_alpha", log_1m_alpha)
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha)
        self.register_buffer("log_1m_cumprod_alpha", log_1m_cumprod_alpha)

        self.register_buffer("Lt_history", torch.zeros(timesteps))
        self.register_buffer("Lt_count", torch.zeros(timesteps))

    def predict_start(
        self, log_xt: torch.Tensor, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Runs the denoiser and returns log softmax predictions."""
        xt = _log_onehot_to_index(log_xt)
        out = self.denoise_fn(t, xt, z)
        log_pred = F.log_softmax(out, dim=1)

        return log_pred

    def predict_reverse_step(
        self, log_x: torch.Tensor, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the model's prediction based on parametrization strategy."""
        if self.parametrization == "x0":
            log_x_recon = self.predict_start(log_x, t=t, z=z)
            log_model_pred = q_posterior(
                log_x0=log_x_recon,
                log_xt=log_x,
                t=t,
                log_alpha=self.log_alpha,
                log_alphabar=self.log_cumprod_alpha,
            )
        else:
            log_model_pred = self.predict_start(log_x, t=t, z=z)

        return log_model_pred

    def q_sample(self, log_x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Samples a noisy sequence at time t from the clean data."""
        log_xt_probs = q_forward(log_x0, t, self.log_cumprod_alpha, self.num_classes)
        log_sample = _log_sample_categorical(log_xt_probs, self.num_classes)

        return log_sample

    def kl_prior(self, log_x0: torch.Tensor) -> torch.Tensor:
        """Calculates the KL divergence prior at the final diffusion step."""
        ones = torch.ones(log_x0.size(0), device=log_x0.device, dtype=torch.long)

        log_final_noise_probs = q_forward(
            log_x0=log_x0,
            t=(self.num_timesteps - 1) * ones,
            log_alphabar=self.log_cumprod_alpha,
            num_classes=self.num_classes,
        )

        log_uniform_prior = -torch.log(
            self.num_classes * torch.ones_like(log_final_noise_probs)
        )
        kl_prior = self.multinomial_kl(log_final_noise_probs, log_uniform_prior)

        return torch.sum(kl_prior, dim=1)

    def sample_time(
        self, batch_size: int, device: torch.device, method: str = "uniform"
    ) -> tuple:
        """Samples a batch of timesteps and their associated probabilities."""
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

            return sampled_timesteps, sampled_probs

        elif method == "uniform":
            sampled_timesteps = torch.randint(
                0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long
            )
            sampled_probs = (
                torch.ones_like(sampled_timesteps).float() / self.num_timesteps
            )

            return sampled_timesteps, sampled_probs

        else:
            raise ValueError(f"Unknown sample time method: {method}")

    def compute_full_vlb(self, log_x0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        log_x0 = _index_to_log_onehot(log_x0, self.num_classes)
        loss = 0
        for t_idx in range(0, self.num_timesteps):
            t_array = torch.full(
                (log_x0.size(0),), t_idx, device=log_x0.device, dtype=torch.long
            )
            log_xt = self.q_sample(log_x0, t_array)
            log_pred = self.predict_reverse_step(log_xt, t_array, z)

            kl = compute_vlb_loss(
                log_x0=log_x0,
                log_xt=log_xt,
                t=t_array,
                log_alpha=self.log_alpha,
                log_alphabar=self.log_cumprod_alpha,
                log_pred=log_pred,
            )
            loss += kl

        loss += self.kl_prior(log_x0)
        return loss

    def _train_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "vb_stochastic":
            sampled_timesteps, sampled_probs = self.sample_time(
                x.size(0), x.device, "importance"
            )
            log_x0 = _index_to_log_onehot(x, self.num_classes)
            log_xt = self.q_sample(log_x_start=log_x0, t=sampled_timesteps)
            log_pred = self.predict_reverse_step(log_xt, sampled_timesteps, z)

            loss = compute_vlb_loss(
                log_x0=log_x0,
                log_xt=log_xt,
                t=sampled_timesteps,
                log_alpha=self.log_alpha,
                log_alphabar=self.log_cumprod_alpha,
                log_pred=log_pred,
            )

            # Update loss history for importance sampling
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
            return -self.compute_full_vlb(x)

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
                log_alphabar=self.log_cumprod_alpha,
                log_pred=log_pred,
            )

            kl_prior = self.kl_prior(log_x0)
            total_loss = (loss / sampled_probs) + kl_prior

            return -total_loss
