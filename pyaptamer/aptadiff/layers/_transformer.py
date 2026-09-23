__author__ = ["aditi-dsi"]
__all__ = ["SinusoidalPosEmbedding", "AptaDiffTransformerEmbedding"]

import math
import warnings

import torch
import torch.nn as nn

from pyaptamer.aptadiff.layers._libs._axial_positional_embedding import (
    AxialPositionalEmbedding,
)
from pyaptamer.aptadiff.layers._libs._linear_attention_transformer import (
    LinearAttentionTransformer,
)


class SinusoidalPosEmbedding(nn.Module):
    """Computes sinusoidal positional embeddings for diffusion timesteps.

    Parameters
    ----------
    dim : int
        Dimension of the positional embedding vector.
    num_steps : int
        Total number of diffusion timesteps in the process.
    rescale_steps : int, default=4000
        Rescaling denominator for timestep normalization.
    """

    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        self.rescale_steps = rescale_steps

    def forward(self, x):
        """Compute sinusoidal embeddings for input timesteps.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of timestep indices of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Sinusoidal timestep embeddings of shape (batch_size, dim).
        """
        x = x * (self.rescale_steps / self.num_steps)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(1) * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class AptaDiffTransformerEmbedding(nn.Module):
    """Transformer-based feature extractor for AptaDiff that combines
    sequence embeddings with diffusion timesteps and latent conditions.

    Supports both native PyTorch scaled dot-product attention and custom
    linear attention implementations.

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
    transformer_type : str, default="native"
        The attention backend to use.
        - "native": Uses PyTorch's optimized nn.TransformerEncoderLayer.
          Recommended for standard aptamer lengths (<= 100nt) as it provides exact
          attention math and maximum hardware acceleration.
        - "linear": Uses the linear attention approximation from the original
          AptaDiff paper. Set to this for strict reproducibility.

    Notes
    -----
    The linear attention implementation is adapted from the original
    AptaDiff codebase and the `lucidrains/linear-attention-transformer` repository
    (https://github.com/lucidrains/linear-attention-transformer).

    References
    ----------
    .. [1] Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. "Transformers
           are RNNs: Fast Autoregressive Transformers with Linear Attention."
           International Conference on Machine Learning (ICML), 2020.
           https://arxiv.org/abs/2006.16236
    """

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
    ):
        super().__init__()

        if transformer_type not in ["linear", "native"]:
            raise ValueError(
                f"transformer_type must be 'linear' or 'native', "
                f"got'{transformer_type}'"
            )

        if transformer_type == "native" and n_local_attn_heads != 0:
            warnings.warn(
                f"n_local_attn_heads is set to {n_local_attn_heads} but "
                "will be ignored because transformer_type is 'native'. "
                "This parameter is only used when transformer_type='linear'.",
                stacklevel=2,
            )

        if max_seq_len % local_attn_window_size != 0:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be evenly divisible by "
                f"local_attn_window_size ({local_attn_window_size}) to calculate "
                "axial embeddings."
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
                if transformer_type == "linear":
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
                elif transformer_type == "native":
                    block.append(
                        nn.TransformerEncoderLayer(
                            d_model=dim,
                            nhead=heads,
                            dim_feedforward=dim * 4,
                            dropout=attn_layer_dropout,
                            activation="gelu",
                            batch_first=True,
                            norm_first=True,
                        )
                    )
            self.transformer_blocks.append(block)

        self.norm = nn.LayerNorm(self.emb_dim)
        self.out = nn.Linear(self.emb_dim, output_dim)

    def forward(self, x, t, z):
        """Pass inputs through the embedding and conditioned transformer layers.

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
            Conditioned embedded sequence tensor of shape
            (batch_size, seq_len, output_dim).
        """
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
