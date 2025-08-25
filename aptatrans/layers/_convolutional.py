__author__ = ["nennomp"]
__all__ = ["ConvBlock"]

from collections import OrderedDict
from collections.abc import Callable

import torch.nn as nn
from torch import Tensor


def conv3x3(in_c: int, out_c: int) -> nn.Conv2d:
    """Initialize a 3x3 2D convolution with padding and bias.

    Parameters
    ----------
    in_c : int
        Number of input channels.
    out_c : int
        Number of output channels.

    Returns
    -------
    nn.Conv2d
        A initialized 3x3 2D convolution layer.
    """
    return nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=3,
        padding="same",
    )


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of (pooling -> conv3x3 -> batchnorm -> GELU)
    layers.

    Parameters
    ----------
    in_c : int
        Number of input channels.
    out_c : int
        Number of output channels.
    pooling : Optional[Callable[..., nn.Module]], optional, default=None
        Instance of a (callable) pooling operator.

    Attributes
    ----------
    block : nn.Sequential
        A sequential container for the block architectural components and activations.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        pooling: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.block = self._init_block(in_c, out_c, pooling)

    def _init_block(
        self,
        in_c: int,
        out_c: int,
        pooling: Callable[..., nn.Module],
    ) -> nn.Sequential:
        """Initialize a convolutional block with pooling.

        Returns
        -------
        nn.Sequential
            A sequential container of layers.
        """
        layers = OrderedDict()

        if pooling is not None:
            layers["pooling"] = pooling

        layers.update(
            [
                ("conv1", conv3x3(in_c, out_c)),
                ("bn1", nn.BatchNorm2d(out_c)),
                ("activation1", nn.GELU()),
                ("conv2", conv3x3(out_c, out_c)),
                ("bn2", nn.BatchNorm2d(out_c)),
                ("activation2", nn.GELU()),
            ]
        )

        return nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, n_channels (`in_c`), height, width).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, n_channels (`out_c`), height, width) if
            no downsamplign occurs, (batch_size, n_channels (`out_c`), height // 2,
            width // 2) otherwise.
        """
        identity = x

        out = self.block(x)
        # residual connection (if no downsampling has occurred, i.e. no pooling)
        if out.shape == identity.shape:
            out += identity

        return out
