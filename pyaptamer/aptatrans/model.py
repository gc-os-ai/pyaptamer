"""Implementation of AptaTrans deep neural network [1][2].

This module implements the transformer-based deep neural network employed within the AptaTrans 
pipeline.

References:
[1] I. Shin et al., AptaTrans: a deep neural network for predicting aptamer-protein interaction using pretrained encoders, BMC Bioinformatics, (2023)
[2] https://github.com/PNUMLB/AptaTrans
"""

__author__ = ["nennomp"]
__all__ = ["AptaTrans"]

from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pyaptamer.aptatrans.layers.convolutional import ConvBlock
from pyaptamer.aptatrans.layers.encoder import EmbeddingConfig, PositionalEncoding, TokenPredictor
from pyaptamer.aptatrans.layers.utils import InteractionMap


class AptaTrans(nn.Module):
    def __init__(
        self, 
        apta_embedding: 'EmbeddingConfig',
        prot_embedding: 'EmbeddingConfig',
        n_encoder_layers: int = 6,
        in_dim: int = 128,
        n_heads: int = 8,
        conv_layers: list[int] = [3, 3, 3],
        dropout: float = 0.1,
    ) -> None:
        """
        apta_embedding: An instance of the EmbeddingConfig() dataclass for storing aptamers' 
            embedding constants.
        prot_embedding: An instance of the EmbeddingConfig() dataclass for storing proteins' 
            embedding constants.
        n_encoder_layers: number of TransformerEncoderLayer() layers.
        in_dim: Input dimension for the embeddings.
        n_heads: Number of heads for attention.
        conv_layers: Number of ConvBlock() in each convolutional layer.
        """
        super().__init__()
        self.apta_embedding = apta_embedding
        self.prot_embedding = prot_embedding

        self.inplanes = 64

        # Encoder for aptamers
        self.encoder_apta = self._make_encoder(
            in_dim=in_dim,
            n_heads=n_heads,
            dropout=dropout,
            n_layers=n_encoder_layers,
            n_vocabs=apta_embedding.n_vocabs,
            n_target_vocabs=apta_embedding.n_target_vocabs,
            max_len=apta_embedding.max_len,
        )
        # Encoder for proteins
        self.encoder_prot = self._make_encoder(
            in_dim=in_dim,
            n_heads=n_heads,
            dropout=dropout,
            n_layers=n_encoder_layers,
            n_vocabs=prot_embedding.n_vocabs,
            n_target_vocabs=prot_embedding.n_target_vocabs,
            max_len=prot_embedding.max_len,
        )
        
        self.imap = InteractionMap()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(planes=64, n_blocks=conv_layers[0])
        self.layer2 = self._make_layer(
            planes=128, 
            n_blocks=conv_layers[1], 
            pooling=nn.MaxPool2d(2, 2)
        )
        self.layer3 = self._make_layer(
            planes=256, 
            n_blocks=conv_layers[2], 
            pooling=nn.MaxPool2d(2, 2)
        )

        # Fully-connected head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            OrderedDict([
                ('linear1', nn.Linear(self.inplanes, self.inplanes // 2)),
                ('activation1', nn.GELU()),
                ('linear2', nn.Linear(self.inplanes // 2, 1)),
                ('activation2', nn.Sigmoid()),
            ])
        )

    def _make_encoder(
        self, 
        in_dim: int, 
        n_heads: int, 
        dropout: float, 
        n_layers: int,
        n_vocabs: int, 
        n_target_vocabs: int,
        max_len: int,
    ) -> nn.Module:
        """Initialize a (transformer-based) encoder consisting of

        (embedding -> positional encoding -> attention -> token predictor)
        """
        embedding = nn.Embedding(
            num_embeddings=n_vocabs, 
            embedding_dim=in_dim, 
            padding_idx=0,
        )
        pos_encoding = PositionalEncoding(d_model=in_dim, max_len=max_len)

        encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=n_heads,
                dim_feedforward=in_dim * 2,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(in_dim),
        )
        token_predictor = TokenPredictor(
            d_model=in_dim,
            n_vocabs=n_vocabs,
            n_target_vocabs=n_target_vocabs,
        )

        return nn.Sequential(
            OrderedDict([
                ('embedding', embedding),
                ('pos_encoding', pos_encoding),
                ('encoder', encoder),
                ('token_predictor', token_predictor),
            ])
        )

    def _make_layer(
        self, 
        planes: int, 
        n_blocks: int, 
        pooling: Optional[Callable[..., nn.Module]] = None,
    ) -> nn.Sequential:
        """Initialize a convolutional layer consisting of multiple ConvBlock() class instances.

        Args:
            planes: Number of output channels.
            pooling: An instance of a pooling operator.
        """
        layers = []
        layers.append(ConvBlock(self.inplanes, planes, pooling=pooling))
        for _ in range(1, n_blocks):
            layers.append(ConvBlock(planes, planes))

        self.inplanes = planes # update input channels for future blocks

        return nn.Sequential(*layers)

    def forward(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        x_apta = self.encoder_apta[:-1](x_apta) # skip token predictor
        x_prot = self.encoder_prot[:-1](x_prot)

        out = self.imap(x_apta, x_prot)

        out = torch.squeeze(out, dim=2) # remove extra dimension
        out = F.gelu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out