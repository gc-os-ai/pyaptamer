# Copyright (c) 2020 Phil Wang (lucidrains).
# Licensed under the MIT license.

"""Linear attention transformer, vendored from lucidrains/linear-attention-transformer.

Source: https://github.com/lucidrains/linear-attention-transformer
See pyaptamer/aptadiff/layers/_libs/README.md and LICENSE for the full
attribution and license text.
"""

__author__ = ["aditi-dsi"]
__all__ = ["LinearAttentionTransformer"]

from functools import partial

import torch
from torch import einsum, nn

from pyaptamer.aptadiff.layers._libs._local_attention import LocalAttention

# helper functions


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    left = (*pre_slices, slice(None, index))
    right = (*pre_slices, slice(index, None))
    return t[left], t[right]


# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


# feedforward


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.w1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.w2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


# self attention layer


def linear_attn(q, k, v):
    dim = q.shape[-1]

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)
    q = q * (dim**-0.5)

    context = einsum("bhnd,bhne->bhde", k, v)
    attn = einsum("bhnd,bhde->bhne", q, context)
    return attn.reshape(*q.shape)


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads, n_local_attn_heads, local_attn_window_size, attn_layer_dropout
    ):
        super().__init__()

        if (dim % heads) != 0:
            raise ValueError("embedding dimension must be divisible by number of heads")

        d_heads = dim // heads

        self.heads = heads
        self.d_heads = d_heads
        self.local_attn_heads = n_local_attn_heads

        self.local_attn = LocalAttention(
            window_size=local_attn_window_size, dropout=0.0
        )

        self.to_q = nn.Linear(dim, d_heads * heads, bias=False)
        self.to_k = nn.Linear(dim, d_heads * heads, bias=False)
        self.to_v = nn.Linear(dim, d_heads * heads, bias=False)

        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(attn_layer_dropout)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        b, t, _, _, dh = *q.shape, self.heads, self.d_heads

        def merge_heads(tensor):
            return tensor.reshape(*tensor.shape[:2], -1, dh).transpose(1, 2)

        q, k, v = map(merge_heads, (q, k, v))

        out = []

        split_index_fn = partial(split_at_index, 1, self.local_attn_heads)
        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

        if lq.shape[1] > 0:
            out.append(self.local_attn(lq, lk, lv))

        if q.shape[1] > 0:
            out.append(linear_attn(q, k, v))

        attn = torch.cat(out, dim=1)
        attn = attn.transpose(1, 2).reshape(b, t, -1)

        return self.dropout(self.to_out(attn))


# transformer classes


class LinearAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        n_local_attn_heads=0,
        local_attn_window_size=128,
        attn_layer_dropout=0.0,
    ):
        super().__init__()

        if type(n_local_attn_heads) is not tuple:
            n_local_attn_heads = tuple([n_local_attn_heads] * depth)

        if len(n_local_attn_heads) != depth:
            raise ValueError(
                "n_local_attn_heads tuple must have the same length as depth"
            )
        if any(local_heads > heads for local_heads in n_local_attn_heads):
            raise ValueError(
                "number of local attn heads must not exceed the total number of heads"
            )

        self.layers = nn.ModuleList([])

        for local_heads in n_local_attn_heads:
            attn = SelfAttention(
                dim=dim,
                heads=heads,
                n_local_attn_heads=local_heads,
                local_attn_window_size=local_attn_window_size,
                attn_layer_dropout=attn_layer_dropout,
            )
            ff = FeedForward(dim)

            self.layers.append(nn.ModuleList([PreNorm(dim, attn), PreNorm(dim, ff)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
