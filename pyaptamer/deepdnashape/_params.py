"""Constants of the pretrained deepDNAshape models.

``BASE_VECTORS`` maps each base to its one-hot vector, in the order the
pretrained weights expect. ``N`` is the average of the four bases.

``FEATURES`` maps each DNA shape feature to its `FeatureParams`.

The values come from the original deepDNAshape project:

https://github.com/JinsenLi/deepDNAshape
"""

from typing import NamedTuple

BASE_VECTORS = {
    "T": [1.0, 0.0, 0.0, 0.0],
    "G": [0.0, 1.0, 0.0, 0.0],
    "C": [0.0, 0.0, 1.0, 0.0],
    "A": [0.0, 0.0, 0.0, 1.0],
    "N": [0.25, 0.25, 0.25, 0.25],
}


class FeatureParams(NamedTuple):
    """Settings of one DNA shape feature.

    ``interbase`` is True for features with one value per step between two
    bases, and False for features with one value per base.

    ``flip_rev`` is True for features whose reverse-strand prediction is
    negated before it is averaged with the forward one.

    The network emits normalized scores. ``score * scale + offset`` converts a
    score back to the physical unit of the feature. ``scale`` and ``offset`` are
    the training-set percentile range and median from the original
    ``params.json``, or its standard deviation and mean for ``EP``.
    """

    interbase: bool
    flip_rev: bool
    scale: float
    offset: float


FEATURES = {
    "Shear": FeatureParams(
        interbase=False,
        flip_rev=True,
        scale=0.66,
        offset=0.0,
    ),
    "Stretch": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=0.09669999999998254,
        offset=-0.03,
    ),
    "Stagger": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=0.51,
        offset=-0.03,
    ),
    "Buckle": FeatureParams(
        interbase=False,
        flip_rev=True,
        scale=19.21339999999998,
        offset=0.0,
    ),
    "ProT": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=18.2867,
        offset=-7.18,
    ),
    "Opening": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=2.93,
        offset=0.19,
    ),
    "MGW": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=3.116666666666667,
        offset=5.01,
    ),
    "EP": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=6.022025377651401,
        offset=-6.782113572103729,
    ),
    "Shear-FL": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=0.10999999999999999,
        offset=0.25,
    ),
    "Stretch-FL": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=0.13,
        offset=0.12,
    ),
    "Stagger-FL": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=0.2,
        offset=0.32,
    ),
    "Buckle-FL": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=7.74,
        offset=7.72,
    ),
    "ProT-FL": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=9.31,
        offset=5.85,
    ),
    "Opening-FL": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=8.28,
        offset=3.47,
    ),
    "MGW-FL": FeatureParams(
        interbase=False,
        flip_rev=False,
        scale=2.08,
        offset=1.03,
    ),
    "Shift": FeatureParams(
        interbase=True,
        flip_rev=True,
        scale=1.0,
        offset=0.0,
    ),
    "Slide": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=1.0599999999999998,
        offset=-1.54,
    ),
    "Rise": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=0.5800000000000001,
        offset=3.37,
    ),
    "Tilt": FeatureParams(
        interbase=True,
        flip_rev=True,
        scale=5.42,
        offset=0.0,
    ),
    "Roll": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=16.71,
        offset=-1.45,
    ),
    "HelT": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=6.91,
        offset=34.49,
    ),
    "Shift-FL": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=0.32999999999999996,
        offset=0.48,
    ),
    "Slide-FL": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=0.26999999999999996,
        offset=0.42,
    ),
    "Rise-FL": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=0.21999999999999997,
        offset=0.27,
    ),
    "Tilt-FL": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=1.7599999999999998,
        offset=3.34,
    ),
    "Roll-FL": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=4.629999999999999,
        offset=5.02,
    ),
    "HelT-FL": FeatureParams(
        interbase=True,
        flip_rev=False,
        scale=1.8700000000000006,
        offset=3.79,
    ),
}
