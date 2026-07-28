__author__ = ["aditi-dsi"]
__all__ = ["SinusoidalPosEmbedding", "LinearAttentionTransformerEmbedding"]

import math

import torch
import torch.nn as nn

from pyaptamer.aptadiff.layers._axial_positional_embedding import (
    AxialPositionalEmbedding,
)
from pyaptamer.aptadiff.layers._linear_attention_transformer import (
    LinearAttentionTransformer,
)


class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        self.rescale_steps = rescale_steps

    def forward(self, x):
        x = x * (self.rescale_steps / self.num_steps)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(1) * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class LinearAttentionTransformerEmbedding(nn.Module):
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
        **kwargs,
    ):
        super().__init__()

        if max_seq_len % local_attn_window_size != 0:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be evenly divisible by "
                f"local_attn_window_size ({local_attn_window_size})."
            )

        self.max_seq_len = max_seq_len
        self.depth = depth
        self.n_blocks = n_blocks
        self.emb_dim = dim

        self.first = nn.Embedding(input_dim, self.emb_dim)
        self.time_pos_emb = SinusoidalPosEmbedding(self.emb_dim, num_timesteps)
        self.z_linear = nn.Linear(enc_embed_size, self.emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 4),
            nn.Softplus(),
            nn.Linear(self.emb_dim * 4, self.emb_dim * n_blocks * depth),
        )

        self.axial_pos_emb = AxialPositionalEmbedding(
            self.emb_dim,
            axial_shape=(max_seq_len // local_attn_window_size, local_attn_window_size),
        )

        self.transformer_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.ModuleList()
            for _ in range(depth):
                block.append(
                    LinearAttentionTransformer(
                        dim=dim,
                        depth=1,
                        heads=heads,
                        n_local_attn_heads=n_local_attn_heads,
                        local_attn_window_size=local_attn_window_size,
                        attn_layer_dropout=attn_layer_dropout,
                    )
                )
            self.transformer_blocks.append(block)

        self.norm = nn.LayerNorm(self.emb_dim)
        self.out = nn.Linear(self.emb_dim, output_dim)

    def forward(self, x, t, z, **kwargs):
        t_emb = self.time_pos_emb(t)
        z_emb = self.z_linear(z)

        cond = t_emb + z_emb
        cond = self.mlp(cond)

        time_embed = cond.view(x.size(0), 1, self.emb_dim, self.n_blocks, self.depth)

        x_emb = self.first(x)
        x_embed_axial = x_emb + self.axial_pos_emb(x_emb).type(x_emb.type())

        h = torch.zeros_like(x_embed_axial)

        for i, block in enumerate(self.transformer_blocks):
            h = h + x_embed_axial
            for j, transformer in enumerate(block):
                h = transformer(h + time_embed[..., i, j])

        h = self.norm(h)
        return self.out(h)
