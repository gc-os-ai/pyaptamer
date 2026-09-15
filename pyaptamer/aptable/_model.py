"""AptaBLE deep neural network for aptamer-protein interaction prediction.

AptaBLE replaces the dot-product InteractionMap in AptaTrans with a symmetric
bidirectional cross-attention layer. Everything else (encoders, convolutional
head, fully-connected output) is identical to AptaTrans, so pretrained
AptaTrans encoder weights are directly reusable.

Reference
---------
AptaBLE: Aptamer Binding and Likelihood Estimation.
NeurIPS 2024 workshop, updated January 2026.
"""

__author__ = ["DZDasherKTB"]
__all__ = ["AptaBLE"]

import os
from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from pyaptamer import logger
from pyaptamer.aptatrans.layers._convolutional import ConvBlock
from pyaptamer.aptatrans.layers._encoder import (
    EncoderPredictorConfig,
    PositionalEncoding,
    TokenPredictor,
)
from pyaptamer.aptable.layers._cross_attention_map import CrossAttentionInteractionMap


class AptaBLE(nn.Module):
    """AptaBLE: AptaTrans with symmetric bidirectional cross-attention.

    Identical to AptaTrans in every respect except the interaction map:
    the dot-product ``InteractionMap`` is replaced with
    ``CrossAttentionInteractionMap``, which allows aptamers and proteins
    to attend to each other's positions simultaneously.

    This produces interaction maps that recapitulate experimental binding
    interfaces. The attention weights returned by ``forward_imap(...,
    return_attention=True)`` serve as interpretable residue-level contact
    predictions.

    Pretrained AptaTrans encoder weights are directly compatible — use
    ``load_aptatrans_encoders()`` to transfer them.

    Parameters
    ----------
    apta_embedding, prot_embedding : EncoderPredictorConfig
        Encoder configuration for aptamers and proteins respectively.
        Must match the config used when pretraining AptaTrans encoders
        if you intend to load pretrained weights.
    in_dim : int, optional, default=128
        Feature dimension of encoder outputs.
    n_encoder_layers : int, optional, default=6
        Number of transformer encoder layers.
    n_heads : int, optional, default=8
        Number of attention heads in the transformer encoders.
    cross_attention_heads : int, optional, default=8
        Number of attention heads in the CrossAttentionInteractionMap.
        Must divide ``in_dim`` evenly.
    dropout : float, optional, default=0.1
        Dropout rate in encoders and cross-attention.
    conv_layers : list[int], optional, default=[3, 3, 3]
        Number of convolutional blocks per layer in the conv head.
    use_layer_norm : bool, optional, default=True
        Apply LayerNorm after cross-attention residual connections.

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.aptable import AptaBLE
    >>> from pyaptamer.aptatrans import EncoderPredictorConfig
    >>> apta_cfg = EncoderPredictorConfig(128, 16, max_len=128)
    >>> prot_cfg = EncoderPredictorConfig(128, 16, max_len=128)
    >>> model = AptaBLE(apta_cfg, prot_cfg, in_dim=32, n_encoder_layers=1,
    ...                 n_heads=4, cross_attention_heads=4, conv_layers=[1,1,1])
    >>> x_apta = torch.randint(1, 16, (2, 10))
    >>> x_prot = torch.randint(1, 16, (2, 12))
    >>> preds = model(x_apta, x_prot)
    >>> preds.shape
    torch.Size([2, 1])
    >>> # Interpretable contact maps
    >>> imap, attn_a2p, attn_p2a = model.forward_imap(
    ...     x_apta, x_prot, return_attention=True
    ... )
    >>> attn_a2p.shape   # (batch, seq_apta, seq_prot)
    torch.Size([2, 10, 12])
    """

    def __init__(
        self,
        apta_embedding: EncoderPredictorConfig,
        prot_embedding: EncoderPredictorConfig,
        in_dim: int = 128,
        n_encoder_layers: int = 6,
        n_heads: int = 8,
        cross_attention_heads: int = 8,
        dropout: float = 0.1,
        conv_layers: list[int] | None = None,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if in_dim % n_heads != 0:
            raise ValueError(
                f"`in_dim` ({in_dim}) must be divisible by `n_heads` ({n_heads})."
            )

        if conv_layers is None:
            conv_layers = [3, 3, 3]

        self.inplanes = 64
        self.apta_embedding = apta_embedding
        self.prot_embedding = prot_embedding

        # Encoders: identical architecture to AptaTrans
        self.encoder_apta, self.token_predictor_apta = self._make_encoder(
            embedding_config=apta_embedding,
            in_dim=in_dim,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.encoder_prot, self.token_predictor_prot = self._make_encoder(
            embedding_config=prot_embedding,
            in_dim=in_dim,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Cross-attention interaction map (the AptaBLE innovation)
        self.imap = CrossAttentionInteractionMap(
            d_model=in_dim,
            n_heads=cross_attention_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # Convolutional head: identical to AptaTrans
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.gelu1 = nn.GELU()
        self.layer1 = self._make_layer(planes=64, n_blocks=conv_layers[0])
        self.layer2 = self._make_layer(
            planes=128, n_blocks=conv_layers[1], pooling=nn.MaxPool2d(2, 2)
        )
        self.layer3 = self._make_layer(
            planes=256, n_blocks=conv_layers[2], pooling=nn.MaxPool2d(2, 2)
        )

        # Fully-connected output head: identical to AptaTrans
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(self.inplanes, self.inplanes // 2)),
                    ("activation1", nn.GELU()),
                    ("linear2", nn.Linear(self.inplanes // 2, 1)),
                    ("activation2", nn.Sigmoid()),
                ]
            )
        )

    # ------------------------------------------------------------------
    # Encoder construction (identical to AptaTrans._make_encoder)
    # ------------------------------------------------------------------

    def _make_encoder(
        self,
        embedding_config: EncoderPredictorConfig,
        in_dim: int,
        n_encoder_layers: int,
        n_heads: int,
        dropout: float,
    ) -> tuple[nn.Module, TokenPredictor]:
        embedding = nn.Embedding(
            num_embeddings=embedding_config.num_embeddings,
            embedding_dim=in_dim,
            padding_idx=0,
        )
        pos_encoding = PositionalEncoding(
            d_model=in_dim, max_len=embedding_config.max_len
        )
        encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=n_heads,
                dim_feedforward=in_dim * 2,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(in_dim),
        )
        token_predictor = TokenPredictor(
            d_model=in_dim,
            d_out_mt=embedding_config.num_embeddings,
            d_out_ss=embedding_config.target_dim,
        )
        encoder_module = nn.ModuleList([embedding, pos_encoding, encoder])
        encoder_module.forward = lambda x: encoder(
            pos_encoding(embedding(x)),
            src_key_padding_mask=(x == 0),
        )
        return (encoder_module, token_predictor)

    def _make_layer(
        self,
        planes: int,
        n_blocks: int,
        pooling: Callable[..., nn.Module] | None = None,
    ) -> nn.Sequential:
        layers = [ConvBlock(self.inplanes, planes, pooling=pooling)]
        for _ in range(1, n_blocks):
            layers.append(ConvBlock(planes, planes))
        self.inplanes = planes
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Pretrained weight transfer from AptaTrans
    # ------------------------------------------------------------------

    def load_aptatrans_encoders(self, aptatrans_state_dict: dict) -> None:
        """Transfer pretrained encoder weights from AptaTrans.

        AptaBLE uses an identical encoder architecture to AptaTrans.
        This method loads only the encoder and token predictor weights,
        leaving the interaction map and convolutional head randomly
        initialised for fine-tuning.

        Parameters
        ----------
        aptatrans_state_dict : dict
            State dict from a trained ``AptaTrans`` model (``model.state_dict()``).

        Examples
        --------
        >>> import torch
        >>> from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig
        >>> from pyaptamer.aptable import AptaBLE
        >>> cfg_a = EncoderPredictorConfig(128, 16, max_len=128)
        >>> cfg_p = EncoderPredictorConfig(128, 16, max_len=128)
        >>> aptatrans = AptaTrans(cfg_a, cfg_p)
        >>> aptable = AptaBLE(cfg_a, cfg_p)
        >>> aptable.load_aptatrans_encoders(aptatrans.state_dict())
        """
        encoder_keys = {
            k: v for k, v in aptatrans_state_dict.items()
            if k.startswith("encoder_apta")
            or k.startswith("encoder_prot")
            or k.startswith("token_predictor_apta")
            or k.startswith("token_predictor_prot")
        }
        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)
        encoder_missing = [k for k in missing if k.startswith("encoder") or k.startswith("token_predictor")]
        if encoder_missing:
            logger.warning(
                "Some encoder keys were not found in the AptaTrans state dict: %s",
                encoder_missing,
            )
        else:
            logger.info(
                "Loaded %d encoder parameter tensors from AptaTrans state dict.",
                len(encoder_keys),
            )

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def forward_encoder(
        self, x: tuple[Tensor, Tensor], encoder_type: str
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through encoder for pretraining (masked token + SS prediction).

        Parameters
        ----------
        x : tuple[Tensor, Tensor]
            (masked_tokens, secondary_structure_tokens). Both shape (batch, seq_len).
        encoder_type : str
            'apta' or 'prot'.

        Returns
        -------
        tuple[Tensor, Tensor]
            (masked_token_predictions, secondary_structure_predictions).

        Raises
        ------
        ValueError
            If encoder_type is not 'apta' or 'prot'.
        """
        if encoder_type == "apta":
            out_mt = self.encoder_apta(x[0])
            out_ss = self.encoder_apta(x[1])
            return self.token_predictor_apta(out_mt, out_ss)
        elif encoder_type == "prot":
            out_mt = self.encoder_prot(x[0])
            out_ss = self.encoder_prot(x[1])
            return self.token_predictor_prot(out_mt, out_ss)
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type!r}. Must be 'apta' or 'prot'."
            )

    def forward_imap(
        self,
        x_apta: Tensor,
        x_prot: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Compute the cross-attention interaction map.

        Parameters
        ----------
        x_apta, x_prot : Tensor
            Token index tensors. Shapes: (batch, seq_apta) and (batch, seq_prot).
            Zero-padded tokens are automatically masked.
        return_attention : bool, optional, default=False
            If True, return attention weight matrices alongside the interaction
            map. These serve as interpretable residue-level contact predictions,
            as demonstrated by AptaBLE on SARS-CoV-2 Spike glycoprotein aptamers.

        Returns
        -------
        Tensor
            Interaction map of shape (batch, 1, seq_apta, seq_prot).
        tuple[Tensor, Tensor, Tensor] (when ``return_attention=True``)
            (imap, attn_a2p, attn_p2a) where:
            - ``attn_a2p``: aptamer attends to protein, shape (batch, seq_apta, seq_prot)
            - ``attn_p2a``: protein attends to aptamer, shape (batch, seq_prot, seq_apta)
        """
        apta_pad_mask = (x_apta == 0)
        prot_pad_mask = (x_prot == 0)

        x_apta_enc = self.encoder_apta(x_apta)
        x_prot_enc = self.encoder_prot(x_prot)

        return self.imap(
            x_apta_enc,
            x_prot_enc,
            apta_key_padding_mask=apta_pad_mask,
            prot_key_padding_mask=prot_pad_mask,
            return_attention=return_attention,
        )

    def forward(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        """Full forward pass for inference and fine-tuning.

        Parameters
        ----------
        x_apta, x_prot : Tensor
            Token index tensors. Shapes: (batch, seq_apta) and (batch, seq_prot).

        Returns
        -------
        Tensor
            Predicted binding probability of shape (batch, 1).
        """
        out = self.forward_imap(x_apta, x_prot)

        out = self.gelu1(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
