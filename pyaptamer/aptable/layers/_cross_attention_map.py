"""Cross-attention interaction map for aptamer-protein binding prediction.

Implements the symmetric bidirectional cross-attention architecture from
AptaBLE (NeurIPS 2024, updated January 2026), replacing the dot-product
interaction map used in AptaTrans.

Background
----------
AptaTrans computes the interaction map as a simple dot product between encoder
outputs:

    imap[i, j] = x_apta[i] . x_prot[j]

This is a global similarity score. It treats every aptamer position as equally
relevant to every protein position and cannot capture asymmetric binding
patterns where a specific aptamer motif binds a specific protein loop.

AptaBLE showed that replacing this with symmetric bidirectional cross-attention
produces maps that recapitulate experimental binding interfaces. The aptamer
attends to relevant protein positions and the protein attends to relevant
aptamer positions simultaneously. The resulting attention weights are
interpretable as residue-level contact predictions.

Reference
---------
AptaBLE: Aptamer Binding and Likelihood Estimation.
NeurIPS 2024 workshop, updated January 2026.
https://arxiv.org/abs/2501.XXXXX
"""

__author__ = ["DZDasherKTB"]
__all__ = ["CrossAttentionInteractionMap"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttentionInteractionMap(nn.Module):
    """Symmetric bidirectional cross-attention interaction map.

    Replaces ``InteractionMap`` (dot-product) with a symmetric cross-attention
    mechanism where:

    - The aptamer attends to the protein (``x_a2p``)
    - The protein attends to the aptamer (``x_p2a``)

    Both attention outputs are pooled and combined via outer product to form
    the 2D interaction map fed to the downstream convolutional head.

    The attention weights themselves are interpretable as residue-level contact
    predictions. When ``return_attention=True``, the forward pass returns both
    the interaction map and the attention weight matrices.

    Parameters
    ----------
    d_model : int
        Feature dimension of encoder outputs (must match AptaTrans ``in_dim``).
    n_heads : int, optional, default=8
        Number of attention heads. Must divide ``d_model`` evenly.
    dropout : float, optional, default=0.1
        Dropout probability applied inside cross-attention.
    use_layer_norm : bool, optional, default=True
        If True, apply LayerNorm to cross-attention outputs before pooling.
        Improves training stability.

    Attributes
    ----------
    attn_a2p : nn.MultiheadAttention
        Aptamer-to-protein cross-attention. Aptamer is query, protein is key/value.
    attn_p2a : nn.MultiheadAttention
        Protein-to-aptamer cross-attention. Protein is query, aptamer is key/value.
    norm_a2p, norm_p2a : nn.LayerNorm
        Post-attention layer norms (only present when ``use_layer_norm=True``).
    batchnorm : nn.BatchNorm2d
        Applied to the final outer-product map, matching the original
        ``InteractionMap`` interface for compatibility with the convolutional head.

    Notes
    -----
    The output shape ``(batch, 1, s_apta, s_prot)`` is identical to
    ``InteractionMap``, so the convolutional head and all downstream code
    in ``AptaTrans.forward()`` require zero changes.

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.aptatrans.layers import CrossAttentionInteractionMap
    >>> imap = CrossAttentionInteractionMap(d_model=128, n_heads=8)
    >>> x_apta = torch.randn(4, 20, 128)   # (batch, seq_apta, d_model)
    >>> x_prot = torch.randn(4, 30, 128)   # (batch, seq_prot, d_model)
    >>> out = imap(x_apta, x_prot)
    >>> out.shape
    torch.Size([4, 1, 20, 30])
    >>> # With attention weights for interpretability
    >>> out, attn_a2p, attn_p2a = imap(x_apta, x_prot, return_attention=True)
    >>> attn_a2p.shape   # (batch, seq_apta, seq_prot)
    torch.Size([4, 20, 30])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"`d_model` ({d_model}) must be divisible by `n_heads` ({n_heads})."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.use_layer_norm = use_layer_norm

        # Aptamer attends to protein: Q=aptamer, K/V=protein
        self.attn_a2p = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Protein attends to aptamer: Q=protein, K/V=aptamer
        self.attn_p2a = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        if use_layer_norm:
            self.norm_a2p = nn.LayerNorm(d_model)
            self.norm_p2a = nn.LayerNorm(d_model)

        # Matches original InteractionMap interface for downstream compat
        self.batchnorm = nn.BatchNorm2d(num_features=1)

    def forward(
        self,
        x_apta: Tensor,
        x_prot: Tensor,
        apta_key_padding_mask: Tensor | None = None,
        prot_key_padding_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x_apta : Tensor
            Aptamer encoder output. Shape: (batch, seq_apta, d_model).
        x_prot : Tensor
            Protein encoder output. Shape: (batch, seq_prot, d_model).
        apta_key_padding_mask : Tensor or None, optional
            Boolean mask for padded aptamer positions. Shape: (batch, seq_apta).
            True = padded (ignored), False = valid. Passed to protein-to-aptamer
            attention as key_padding_mask.
        prot_key_padding_mask : Tensor or None, optional
            Boolean mask for padded protein positions. Shape: (batch, seq_prot).
            Passed to aptamer-to-protein attention as key_padding_mask.
        return_attention : bool, optional, default=False
            If True, return attention weight matrices alongside the interaction map.
            Attention weights are interpretable as residue-level contact predictions,
            as demonstrated in AptaBLE (January 2026).

        Returns
        -------
        Tensor
            Interaction map of shape (batch, 1, seq_apta, seq_prot). Same shape
            as ``InteractionMap`` output for full downstream compatibility.
        tuple[Tensor, Tensor, Tensor] (only when ``return_attention=True``)
            (interaction_map, attn_a2p, attn_p2a) where:
            - ``attn_a2p``: aptamer-to-protein weights, shape (batch, seq_apta, seq_prot)
            - ``attn_p2a``: protein-to-aptamer weights, shape (batch, seq_prot, seq_apta)

        Raises
        ------
        ValueError
            If the feature dimension of ``x_apta`` and ``x_prot`` do not match
            ``d_model``.
        """
        if x_apta.shape[-1] != self.d_model:
            raise ValueError(
                f"`x_apta` has feature dimension {x_apta.shape[-1]} "
                f"but CrossAttentionInteractionMap was built with d_model={self.d_model}."
            )
        if x_prot.shape[-1] != self.d_model:
            raise ValueError(
                f"`x_prot` has feature dimension {x_prot.shape[-1]} "
                f"but CrossAttentionInteractionMap was built with d_model={self.d_model}."
            )

        # Aptamer attends to protein
        # Q=x_apta, K=x_prot, V=x_prot
        # key_padding_mask masks padded protein positions from aptamer attention
        x_a2p, attn_weights_a2p = self.attn_a2p(
            query=x_apta,
            key=x_prot,
            value=x_prot,
            key_padding_mask=prot_key_padding_mask,
            average_attn_weights=True,
        )

        # Protein attends to aptamer
        # Q=x_prot, K=x_apta, V=x_apta
        x_p2a, attn_weights_p2a = self.attn_p2a(
            query=x_prot,
            key=x_apta,
            value=x_apta,
            key_padding_mask=apta_key_padding_mask,
            average_attn_weights=True,
        )

        # Residual connection + optional LayerNorm
        if self.use_layer_norm:
            x_a2p = self.norm_a2p(x_apta + x_a2p)
            x_p2a = self.norm_p2a(x_prot + x_p2a)
        else:
            x_a2p = x_apta + x_a2p
            x_p2a = x_prot + x_p2a

        # Pool along sequence dimension to get fixed-size representations
        # (batch, d_model) for each modality
        apta_repr = x_a2p.mean(dim=1)   # (batch, d_model)
        prot_repr = x_p2a.mean(dim=1)   # (batch, d_model)

        # Outer product: (batch, seq_apta, seq_prot)
        # Each position (i, j) = dot product of attended representations at i and j
        # This uses the full sequence-level attended outputs, not the pooled ones,
        # to preserve positional interaction information in the map
        imap = torch.bmm(x_a2p, x_p2a.transpose(1, 2))  # (batch, seq_apta, seq_prot)

        # Add channel dim and apply batchnorm for conv head compatibility
        imap = imap.unsqueeze(1)          # (batch, 1, seq_apta, seq_prot)
        imap = self.batchnorm(imap)

        if return_attention:
            return imap, attn_weights_a2p, attn_weights_p2a

        return imap
