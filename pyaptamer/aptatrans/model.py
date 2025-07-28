"""AptaTrans' deep neural network for aptamer-protein interaction prediction."""

__author__ = ["nennomp"]
__all__ = ["AptaTrans"]

from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from pyaptamer.aptatrans.layers._convolutional import ConvBlock
from pyaptamer.aptatrans.layers._encoder import (
    EncoderPredictorConfig,
    PositionalEncoding,
    TokenPredictor,
)
from pyaptamer.aptatrans.layers._interaction_map import InteractionMap


class AptaTrans(nn.Module):
    """AptaTrans deep neural network as described in [1]_.

    Original implementation:
    - https://github.com/PNUMLB/AptaTrans

    Parameters
    ----------
    apta_embedding : EncoderPredictorConfig
        Instance of the EncoderPredictorConfig() class, containing hyperparameters
        related to the aptamer embeddings.
    prot_embedding : EncoderPredictorConfig
        Instance of the EmbeedingConfig() class, containing hyperparameters related
        to the protein embeddings.
    in_dim : int, optional
        Number of expected input features.
    n_encoder_layers : int, optional
        Number of layers in the encoders.
    n_heads : int, optional
        Number of attention heads in the encoders.
    dropout : float, optional
        Dropout rate for the encoders.
    conv_layers : list[int], optional
        List specifying the number of convolutional blocks in each convolutional
        layer.

    Attributes
    ----------
    inplanes : int
        Number of input channels for the first convolutional layer.
    encoder_apta, encoder_prot : nn.Sequential
        Sequential container of (embedding -> positional encoding -> transformer ->
        token predictor) layers for aptamers and proteins, respectively.
    imap : InteractionMap
        An instance of the InteractionMap() class, for computing the interaction map
        between aptamers and proteins.
    conv1: nn.Conv2d
        First convolutional layer.
    bn1: nn.BatchNorm2d
        Batch normalization layer applied after the first convolutional layer.
    layer1, layer2, layer3 : nn.Sequential
        Sequential containers of ConvBlock() instances.
    avgpool : nn.AdaptiveAvgPool2d
        Adaptive average pooling layer applied before the fully-connected head.
    fc : nn.Sequential
        A sequential container of linear layers and activations for outputting the
        final predictions.

    References
    ----------
    .. [1] Shin, Incheol, et al. "AptaTrans: a deep neural network for predicting
    aptamer-protein interaction using pretrained encoders." BMC bioinformatics 24.1
    (2023): 447.

    Examples
    --------
    >>> from pyaptamer.aptatrans import AptaTrans
    >>> from pyaptamer.aptatrans import EncoderPredictorConfig
    >>> apta_embedding = EncoderPredictorConfig(16, 16, max_len=32)
    >>> prot_embedding = EncoderPredictorConfig(16, 16, max_len=32)
    >>> aptatrans = AptaTrans(apta_embedding, prot_embedding)
    >>> preds = aptatrans(x_apta, x_prot)
    """

    def __init__(
        self,
        apta_embedding: EncoderPredictorConfig,
        prot_embedding: EncoderPredictorConfig,
        in_dim: int = 128,
        n_encoder_layers: int = 6,
        n_heads: int = 8,
        conv_layers: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        apta_embedding : EncoderPredictorConfig
            Instance of the EncoderPredictorConfig() class, containing hyperparameters
            related to the aptamer embeddings.
        prot_embedding : EncoderPredictorConfig
            Instance of the EmbeedingConfig() class, containing hyperparameters related
            to the protein embeddings.
        in_dim : int, optional
            Number of expected input features.
        n_encoder_layers : int, optional
            Number of layers in the encoders.
        n_heads : int, optional
            Number of attention heads in the encoders.
        dropout : float, optional
            Dropout rate for the encoders.
        conv_layers : list[int], optional
            List specifying the number of convolutional blocks in each convolutional
            layer.

        Raises
        AssertionError
            If the input dimension is not divisible by the number of heads.
        """
        super().__init__()
        if in_dim % n_heads != 0:
            raise AssertionError(
                f"Input dimension {in_dim} must be divisible by number of heads "
                f"{n_heads}."
            )
        if conv_layers is None:
            conv_layers = [3, 3, 3]

        self.inplanes = 64
        self.apta_embedding = apta_embedding
        self.prot_embedding = prot_embedding

        # encoder for aptamers
        self.encoder_apta = self._make_encoder(
            embedding_config=apta_embedding,
            in_dim=in_dim,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        # encoder for proteins
        self.encoder_prot = self._make_encoder(
            embedding_config=prot_embedding,
            in_dim=in_dim,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # interaction map
        self.imap = InteractionMap()

        # convolutional layers
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

        # fully-connected head
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

    def _make_encoder(
        self,
        embedding_config: EncoderPredictorConfig,
        in_dim: int,
        n_encoder_layers: int,
        n_heads: int,
        dropout: float,
    ) -> nn.Sequential:
        """
        Initialize a (transformer-based) encoder consisting of (embedding -> positional
        encoding -> transformer -> token predictor) layers.

        Returns
        -------
        nn.Sequential
            A sequential container with the encoder's architectural components.
        """
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

        return nn.Sequential(
            OrderedDict(
                [
                    ("embedding", embedding),
                    ("pos_encoding", pos_encoding),
                    ("encoder", encoder),
                    ("token_predictor", token_predictor),
                ]
            )
        )

    def _make_layer(
        self,
        planes: int,
        n_blocks: int,
        pooling: Callable[..., nn.Module] | None = None,
    ) -> nn.Sequential:
        """Initialize a convolutional layer consisting of multiple ConvBlock()
        instances.

        Parameters
        ----------
        planes : int
            Number of output channels for the convolutional blocks.
        n_blocks : int
            Number of ConvBlock() instances.
        pooling : Optional[Callable[..., nn.Module]], optional
            Pooling layer to apply after the first ConvBlock().

        Returns
        -------
        nn.Sequential
            A sequential container of ConvBlock() instances.
        """
        layers = []
        layers.append(ConvBlock(self.inplanes, planes, pooling=pooling))
        for _ in range(1, n_blocks):
            layers.append(ConvBlock(planes, planes))

        self.inplanes = planes  # update input channels for future blocks

        return nn.Sequential(*layers)

    def forward(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x_apta, x_prot : Tensor
            Input tensors for aptamers and proteins, respectively. Shapes are
            (batch_size, seq_len (s1), n_features) and (batch_size, seq_len (s2),
            n_features), respectively.

        Returns
        -------
        Tensor
            Output tensor of shape (batch__size, 1) containing the model's predictions.
        """
        x_apta = self.encoder_apta[:-1](x_apta)
        x_prot = self.encoder_prot[:-1](x_prot)

        out = self.imap(x_apta, x_prot)

        out = torch.squeeze(out, dim=2)  # remove extra dimension
        out = self.gelu1(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
