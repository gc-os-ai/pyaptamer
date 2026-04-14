"""Feature encoding of strings."""

from pyaptamer.trafos.encode._fcs import FCSWordTransformer
from pyaptamer.trafos.encode._greedy import GreedyEncoder

__all__ = ["GreedyEncoder", "FCSWordTransformer"]
