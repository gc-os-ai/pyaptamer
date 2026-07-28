__author__ = ["lucidrains"]
__all__ = ["LocalAttention"]

import torch
import torch.nn.functional as F
from torch import einsum, nn
from torch.nn import Module

# helper functions


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = padded_x.unfold(1, forward + backward + 1, 1)
    return tensors.movedim(-1, dim).flatten(dim, dim + 1)


# main class


class LocalAttention(Module):
    def __init__(self, window_size, dropout=0.0):
        super().__init__()
        self.window_size = window_size
        self.look_backward = 1
        self.look_forward = 1

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, H, n, dim_head = q.shape
        b_flat = B * H

        q = q.reshape(b_flat, n, dim_head)
        k = k.reshape(b_flat, n, dim_head)
        v = v.reshape(b_flat, n, dim_head)

        scale = dim_head**-0.5
        window_size = self.window_size
        windows = n // window_size

        seq = torch.arange(n, device=q.device)
        b_t = seq.view(1, windows, window_size)

        bq, bk, bv = (t.view(b_flat, windows, window_size, dim_head) for t in (q, k, v))

        bq = bq * scale

        look_around_kwargs = {
            "backward": self.look_backward,
            "forward": self.look_forward,
            "pad_value": -1,
        }

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_k = look_around(b_t, **look_around_kwargs)

        bq_k = bq_k.unsqueeze(-2)

        pad_mask = bq_k == -1

        sim = einsum("b w i e, b w j e -> b w i j", bq, bk)

        mask_value = max_neg_value(sim)
        sim = sim.masked_fill(pad_mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b w i j, b w j e -> b w i e", attn, bv)
        out = out.reshape(b_flat, n, dim_head)

        return out.reshape(B, H, n, dim_head)
