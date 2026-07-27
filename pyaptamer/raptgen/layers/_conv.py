"""Convolutional layers(Bottleneck) for RaptGen's variational autoencoder"""

__author__ = ["NoorMajdoub"]
__all__ = ["Bottleneck"]

from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    """ 
    1D convolutional residual bottleneck block.

A residual block used in the RaptGen CNN encoder for feature extraction, projects the input into higher 
dimension space via a 1x1 convolution layer then captures the 
features with a 7x7 convolution layer (7x7 by default, configurable via window_size) then project back down into the initial dimension.
    Parameters
    ----------
    init_dim :  int, optional, default=32 
    Number of input/output channels. The intermediate convolutions use init_dim * 2 channels.
    
    window_size : int, optional, default=7
    Kernel size for the middle convolution used for feature extraction. Must be odd, so that padding (window_size // 2) keeps the sequence length unchanged.


    Attributes
    ----------
    conv1 : nn.Conv1d 
        1x1 convolution expanding init_dim to init_dim * 2 channels. 
    conv2 : nn.Conv1d 
        Window_size-wide convolution over the expanded channels, adds padding to preserve sequence length. 
    conv3 : nn.Conv1d 
        1x1 convolution projecting back down to init_dim channels. 
    bn1, bn2, bn3 : nn.BatchNorm1d 
        Batch normalization applied before each convolution
    
    """

    def __init__(self, init_dim=32, window_size=7):
        super().__init__()
        assert window_size % 2 == 1, f"window size should be odd, given {window_size}"

        self.conv1 = nn.Conv1d(
            in_channels=init_dim, out_channels=init_dim * 2, kernel_size=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=init_dim * 2,
            out_channels=init_dim * 2,
            kernel_size=window_size,
            padding=window_size // 2,
        )

        self.conv3 = nn.Conv1d(
            in_channels=init_dim * 2, out_channels=init_dim, kernel_size=1
        )

        self.bn1 = nn.BatchNorm1d(init_dim)
        self.bn2 = nn.BatchNorm1d(init_dim * 2)
        self.bn3 = nn.BatchNorm1d(init_dim * 2)

    def forward(self, input):
        x = self.conv1(F.leaky_relu(self.bn1(input)))
        x = self.conv2(F.leaky_relu(self.bn2(x)))
        x = self.conv3(F.leaky_relu(self.bn3(x)))
        return F.leaky_relu(x + input)
