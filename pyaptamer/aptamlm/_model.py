"""
AptaMLM dual-encoder model for protein-conditioned aptamer generation.
"""

__author__ = ["NoorMajdoub"]
__all__ = ["ProteinModule", "NucleotideModule", "AptaMLM"]

import torch
import torch.nn as nn
from torch import Tensor


class ProteinModule(nn.Module):
    """Protein encoder module for AptaMLM.

    Takes frozen ESM2 hidden states and refines them via self-attention,
    then projects to a shared model dimension used by the NucleotideModule.

    Parameters
    ----------
    prot_dim : int, optional, default=1280
        Hidden dimension of the upstream ESM2 encoder.
    d_model : int, optional, default=128
        Internal model dimension.
    n_heads : int, optional, default=4
        Number of attention heads. Must evenly divide ``d_model``.
    n_layers : int, optional, default=3
        Number of stacked self-attention + feedforward layers.
    dropout : float, optional, default=0.1
        Dropout probability applied inside the transformer layers.
    """

    def __init__(
        self,
        prot_dim: int = 1280,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(prot_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, src_key_padding_mask=padding_mask)
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer bridging the protein and nucleotide modules."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nuc_repr: Tensor,
        prot_repr: Tensor,
        prot_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out, _ = self.attn(
            query=nuc_repr,
            key=prot_repr,
            value=prot_repr,
            key_padding_mask=prot_key_padding_mask,
        )
        return self.norm(nuc_repr + self.dropout(attn_out))


class NucleotideModule(nn.Module):
    """Nucleotide encoder module for AptaMLM.

    Takes frozen NucleotideTransformer hidden states, refines them via
    self-attention, then attends to the protein representation via
    cross-attention.

    Parameters
    ----------
    nuc_dim : int, optional, default=1280
        Hidden dimension of the upstream NucleotideTransformer encoder.
    d_model : int, optional, default=128
        Internal model dimension. Must match ``ProteinModule.d_model``.
    n_heads : int, optional, default=4
        Number of attention heads. Must evenly divide ``d_model``.
    n_layers : int, optional, default=3
        Number of stacked self-attention layers applied before cross-attention.
    dropout : float, optional, default=0.1
        Dropout probability.
    """

    def __init__(
        self,
        nuc_dim: int = 1280,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(nuc_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.self_attn_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionLayer(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        nuc: Tensor,
        prot: Tensor,
        nuc_padding_mask: Tensor | None = None,
        prot_padding_mask: Tensor | None = None,
    ) -> Tensor:
        nuc = self.proj(nuc)
        for self_attn, cross_attn in zip(
            self.self_attn_blocks, self.cross_attn_blocks
        ):
            nuc = self_attn(nuc, src_key_padding_mask=nuc_padding_mask)
            nuc = cross_attn(nuc, prot, prot_key_padding_mask=prot_padding_mask)
        return nuc


class AptaMLM(nn.Module):
    """Dual-encoder masked language model for aptamer generation.

    Given a target protein sequence and a (partially masked) aptamer
    sequence, predicts the identity of each masked nucleotide token.

    Both frozen encoders (ESM2 and NucleotideTransformer) are kept fixed
    during training so that only the lightweight bridge layers are updated.
    This is appropriate given the small size of available protein-aptamer
    binding datasets (~725 positive pairs in AptaTrans).

    Parameters
    ----------
    prot_dim : int, optional, default=1280
        Hidden dimension of ESM2 (650M variant outputs 1280).
    nuc_dim : int, optional, default=1280
        Hidden dimension of NucleotideTransformer.
    d_model : int, optional, default=128
        Shared internal dimension for both modules.
    n_heads : int, optional, default=4
        Number of attention heads. Must evenly divide ``d_model``.
    n_layers : int, optional, default=3
        Number of transformer layers in each module.
    dropout : float, optional, default=0.1
        Dropout probability.
    vocab_size : int
        Vocabulary size of the NucleotideTransformer tokenizer.
    """

    def __init__(
        self,
        prot_dim: int = 1280,
        nuc_dim: int = 1280,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        vocab_size: int = 4107,
    ):
        super().__init__()
        self.protein_module = ProteinModule(
            prot_dim, d_model, n_heads, n_layers, dropout
        )
        self.nuc_module = NucleotideModule(
            nuc_dim, d_model, n_heads, n_layers, dropout
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        prot_hidden: Tensor,
        nuc_hidden: Tensor,
        prot_padding_mask: Tensor | None = None,
        nuc_padding_mask: Tensor | None = None,
    ) -> Tensor:
        prot_repr = self.protein_module(prot_hidden, prot_padding_mask)
        nuc_repr = self.nuc_module(
            nuc_hidden, prot_repr, nuc_padding_mask, prot_padding_mask
        )
        return self.head(nuc_repr)


def mask_tokens(
    input_ids: Tensor,
    mask_token_id: int,
    mask_prob: float = 0.15,
) -> tuple[Tensor, Tensor]:
    """Randomly mask a fraction of nucleotide tokens for MLM training.

    Parameters
    ----------
    input_ids : Tensor
        Token ids of shape ``(batch, seq_len)``.
    mask_token_id : int
        The id of the ``[MASK]`` token in the nucleotide tokenizer vocabulary.
    mask_prob : float, optional, default=0.15
        Fraction of tokens to mask.

    Returns
    -------
    masked_input_ids : Tensor
        Token ids with masked positions replaced, same shape as ``input_ids``.
    labels : Tensor
        Ground-truth token ids at masked positions, ``-100`` elsewhere.
    """
    labels = input_ids.clone()
    prob_matrix = torch.full(input_ids.shape, mask_prob)
    mask = torch.bernoulli(prob_matrix).bool()
    labels[~mask] = -100
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask] = mask_token_id
    return masked_input_ids, labels