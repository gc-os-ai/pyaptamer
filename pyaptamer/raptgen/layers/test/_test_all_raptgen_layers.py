


import pytest
import torch
import torch.nn as nn
from torch import Tensor




@pytest.mark.parametrize("in_c, out_c", [(8, 16), (16, 32), (32, 64)])
def test_conv3x3(in_c, out_c):
    """Check correct initialization of a 3x3 convolution layer."""
    conv = conv3x3(in_c, out_c)
    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == in_c
    assert conv.out_channels == out_c
    assert conv.kernel_size == (3, 3)
    assert conv.padding == "same"
