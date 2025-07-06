__author__ = ["nennomp"]
__all__ = ["ConvBlock"]

from collections import OrderedDict
from typing import Callable, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def conv3x3(in_c: int, out_c: int) -> nn.Conv2d:
    """3x3 2D convolution with padding and bias."""
    return nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=3,
        padding='same',
    )


class ConvBlock(nn.Module):
    # Stylistically, using nn.Sequential and OrderedDict is more consistent with what has been 
    # used in other modules of AptaTrans. Additionally, it's also better for extendability and 
    # visualizing the architecture layers..
    def __init__(
        self, 
        in_c: int, 
        out_c: int, 
        pooling: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.block = self._init_block(in_c, out_c, pooling)

    def _init_block(self, in_c, out_c, pooling: Callable[..., nn.Module]) -> nn.Sequential:
        layers = OrderedDict()
        
        if pooling is not None:
            layers['pooling'] = pooling
        layers.update([
            ('conv1', conv3x3(in_c, out_c)),
            ('bn1', nn.BatchNorm2d(out_c)),
            ('activation1', nn.GELU()),
            ('conv2', conv3x3(out_c, out_c)),
            ('activation2', nn.GELU()),
            ('bn2', nn.BatchNorm2d(out_c)),
        ])
        
        return nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.block(x)
        # Residual connection if there has not been downsampling (due to pooling)
        if out.shape == identity.shape:
            out += identity

        return out