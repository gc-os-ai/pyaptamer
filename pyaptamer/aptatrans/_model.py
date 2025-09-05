"""AptaTrans' deep neural network for aptamer-protein interaction prediction."""

__author__ = ["nennomp"]
__all__ = ["AptaTrans"]

import os
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

    Original implementation: https://github.com/PNUMLB/AptaTrans.

    The token predictors in the encoders are needed only for pre-training the encoders
    on masked token and secondary structure prediction, for aptamers and proteins
    respectively. These are not used during fine-tuning or inference.

    Parameters
    ----------
    apta_embedding, prot_embedding : EncoderPredictorConfig
        Instance of the EncoderPredictorConfig() class, containing hyperparameters
        related to the embeddings of aptameters and proteins, respectively.
    in_dim : int, optional, default=128
        Number of expected input features.
    n_encoder_layers : int, optional, default=6
        Number of layers in the encoders.
    n_heads : int, optional, default=8
        Number of attention heads in the encoders.
    dropout : float, optional, default=0.1
        Dropout rate for the encoders.
    conv_layers : list[int], optional, default=[3, 3, 3]
        List specifying the number of convolutional blocks in each convolutional
        layer.
    pretrained : bool, optional, default=False
        If True, load the best weights from the pretrained model.

    Attributes
    ----------
    inplanes : int
        Number of input channels for the first convolutional layer.
    encoder_apta, encoder_prot : nn.Sequential
        Sequential container of (embedding -> positional encoding -> transformer)
        layers for aptamers and proteins, respectively.
    token_predictor_apta, token_predictor_prot : TokenPredictor
        Token predictor layers for aptamers and proteins, respectively. Only used
        during during pre-training to predict masked toklens and secondary structure.
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
    >>> import torch
    >>> from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig
    >>> apta_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> prot_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    >>> x_apta = torch.randint(high=16, size=(128, 10))
    >>> x_prot = torch.randint(high=16, size=(128, 10))
    >>> model = AptaTrans(apta_embedding, prot_embedding, pretrained=False)
    >>> imap = model.forward_imap(x_apta, x_prot)
    >>> preds = model(x_apta, x_prot)
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
        pretrained: bool = False,
    ) -> None:
        """
        Raises
        -------
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

        # encoder and token predictor for aptamers
        self.encoder_apta, self.token_predictor_apta = self._make_encoder(
            embedding_config=apta_embedding,
            in_dim=in_dim,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        # encoder and token predictor for proteins
        self.encoder_prot, self.token_predictor_prot = self._make_encoder(
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

        if pretrained:
            self.load_pretrained_weights()

    def _make_encoder(
        self,
        embedding_config: EncoderPredictorConfig,
        in_dim: int,
        n_encoder_layers: int,
        n_heads: int,
        dropout: float,
    ) -> tuple[nn.Sequential, TokenPredictor]:
        """
        Initialize a (transformer-based) encoder consisting of (embedding -> positional
        encoding -> transformer) layers and a corresponding token predictor for masked
        token and secondary structure prediction.

        Returns
        -------
        nn.Sequential
            A sequential container with the encoder's architectural components.
        TokenPredictor
            A token predictor layer for masked token and secondary structure prediction.
        """
        # transformer-based encoder
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

        # token predictor
        token_predictor = TokenPredictor(
            d_model=in_dim,
            d_out_mt=embedding_config.num_embeddings,
            d_out_ss=embedding_config.target_dim,
        )

        return (
            nn.Sequential(
                OrderedDict(
                    [
                        ("embedding", embedding),
                        ("pos_encoding", pos_encoding),
                        ("encoder", encoder),
                    ]
                )
            ),
            token_predictor,
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
        pooling : Optional[Callable[..., nn.Module]], optional, default=None
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

    def load_pretrained_weights(self, store: bool = True) -> None:
        """Load pretrained model weights from hugging face.

        If the weights are not found locally, they will be downloaded from hugging face.

        Parameters
        ----------
        store : bool, optional, default=True
            If True, the pretrained weights will be saved locally. If False, the weights
            will be downloaded but not saved to disk.
        """
        path = os.path.relpath(
            os.path.join(os.path.dirname(__file__), ".", "weights", "pretrained.pt")
        )

        if os.path.exists(path):
            print(f"Loading pretrained weights from {path}...")
            state_dict = torch.load(path, map_location=torch.device("cpu"))
        else:
            print("Downloading best weights from hugging face...")
            url = (
                "https://huggingface.co/gcos/pyaptamer-aptatrans/resolve/main/"
                "pretrained.pt"
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url=url,
                map_location=torch.device("cpu"),
            )
            print(f"Saving to {path}...")
            torch.save(state_dict, path)

        self.load_state_dict(state_dict, strict=True)

    def forward_encoders(
        self,
        x_apta: tuple[Tensor, Tensor],
        x_prot: tuple[Tensor, Tensor],
    ):
        """Forward pass through the encoders only.

        This method performs a forward pass through the encoders, including the
        token predictors, for pretraining.

        Parameters
        ----------
        x_apta, x_prot : tuple[Tensor, Tensor]
            A tuple of tensors containing the features for masked tokens and secondary
            structure prediction, for aptamers and proteins, respectively. Shapes are
            (batch_size (b1), seq_len (s1)) and (batch_size (b2), seq_len (s2)),
            respectively.

        Returns
        -------
        tuple[Tensor, Tensor], tuple[Tensor, Tensor]
            A tuple of tensors containing the predictions for masked tokens and
            secondary structure, for aptamers and proteins, respectively. For aptamers,
            the shapes are (b1, s1, n_embeddings (n1)) and (b1, s1, target_dim (t1)),
            for the masked tokens and secondary structure, respectively. For proteins,
            the shapes are (b2, s2, n_embeddings (n2)) and (b2, s2, target_dim (t2)),
            respectively.
        """
        # pretrain aptamers' encoder
        out_apta_mt = self.encoder_apta(x_apta[0])
        out_apta_ss = self.encoder_apta(x_apta[1])
        y_apta_mt, y_apta_ss = self.token_predictor_apta(out_apta_mt, out_apta_ss)

        # pretrain proteins' encoder
        out_prot_mt = self.encoder_prot(x_prot[0])
        out_prot_ss = self.encoder_prot(x_prot[1])
        y_prot_mt, y_prot_ss = self.token_predictor_prot(out_prot_mt, out_prot_ss)

        return (y_apta_mt, y_apta_ss), (y_prot_mt, y_prot_ss)

    def forward_imap(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        """Forward pass to compute the interaction map.

        This methods performs a forward pass through the encoders, minus the token
        predictors, to compute the interaction map between aptamers and proteins.

        Parameters
        ----------
        x_apta, x_prot : Tensor
            Input tensors for aptamers and proteins, respectively. Shapes are
            (batch_size, seq_len (s1)) and (batch_size, seq_len (s2)), respectively.

        Returns
        -------
        Tensor
            Interaction map tensor of shape (batch_size, 1, seq_len (s1), seq_len (s2)).
        """
        x_apta, x_prot = self.encoder_apta(x_apta), self.encoder_prot(x_prot)
        return self.imap(x_apta, x_prot)

    def forward(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        """Forward pass.

        This methods performs a forward pass through the entire neural network, minus
        the token predictors, to perform inference and/or fine-tuning.

        Parameters
        ----------
        x_apta, x_prot : Tensor
            Input tensors for aptamers and proteins, respectively. Shapes are
            (batch_size, seq_len (s1)) and (batch_size, seq_len (s2)), respectively.

        Returns
        -------
        Tensor
            Output tensor of shape (batch__size, 1) containing the model's predictions.
        """
        out = self.forward_imap(x_apta, x_prot)

        out = torch.squeeze(out, dim=2)  # remove extra dimension
        out = self.gelu1(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
