__author__ = ["aditi-dsi"]
__all__ = ["Rezero", "AptaDiffDenoiser", "AptaDiffDiffusion"]


import torch
import torch.nn as nn

from pyaptamer.aptadiff._functional import cosine_alpha_schedule, log_one_minus_exp
from pyaptamer.aptadiff.layers._transformer import AptaDiffTransformerEmbedding


def _log_onehot_to_index():
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
        denoise_fn,
        num_classes=4,
        seq_len=...,
        timesteps=1000,
        loss_type="vb_stochastic",
        parametrization="x0",
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

        log_1_min_alpha = log_one_minus_exp(log_alpha)
        log_1_min_cumprod_alpha = log_one_minus_exp(log_cumprod_alpha)

        self.register_buffer("log_alpha", log_alpha)
        self.register_buffer("log_1_min_alpha", log_1_min_alpha)
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha)
        self.register_buffer("log_1_min_cumprod_alpha", log_1_min_cumprod_alpha)

        self.register_buffer("Lt_history", torch.zeros(timesteps))
        self.register_buffer("Lt_count", torch.zeros(timesteps))
