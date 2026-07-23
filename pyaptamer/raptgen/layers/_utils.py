"""Utility functions and classes for RaptGen layers"""

__author__ = ["NoorMajdoub"]
__all__ = ["View", "nt_index", "State", "Transition"]

from enum import IntEnum

from torch import nn


def one_hot_index(seq):
    return [int(nt_index[char]) for char in seq]


class nt_index(IntEnum):  # noqa: N801
    A = 0
    T = 1
    G = 2
    C = 3
    PAD = 4
    SOS = 5
    EOS = 6
    U = 1


class State(IntEnum):
    M = 0
    I = 1  # noqa: E741
    D = 2


class Transition(IntEnum):
    M2M = 0
    M2I = 1
    M2D = 2
    I2M = 3
    I2I = 4
    D2M = 5
    D2D = 6


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = (shape,)

    def forward(self, x):
        return x.view(*self.shape)
