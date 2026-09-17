"""Tests for the PSeAAC transformer."""

__author__ = ["satvshr", "siddharth7113"]

import warnings

import numpy as np
import pandas as pd
import pytest

from pyaptamer.trafos.encode import PSeAAC
from pyaptamer.trafos.encode._pseaac_props import aa_props
from pyaptamer.trafos.encode.tests._pseaac_solution import solution

SEQ = "ACDFFKKIIKKLLMMNNPPQQQRRRRIIIIRRR"


def _frame(*seqs):
    return pd.DataFrame({"protein": list(seqs)})


def test_normalized_values():
    """aa_props(normalize=True) equals the z-scored raw matrix rounded to 3 places."""
    original = aa_props(normalize=False)
    normalized = aa_props(normalize=True)
    manual = ((original - original.mean()) / original.std(ddof=0)).round(3)
    pd.testing.assert_frame_equal(normalized, manual)


@pytest.mark.parametrize(
    "seq,lambda_val",
    [("ACDEFGHIK", 10), ("ACDAA", 5)],
)
def test_sequence_too_short(seq, lambda_val):
    """A sequence not longer than lambda_val raises."""
    with pytest.raises(ValueError, match="Protein sequence is too short"):
        PSeAAC(lambda_val=lambda_val).fit_transform(_frame(seq))


def test_default_matches_aptanet_solution():
    """Default parameters reproduce the AptaNet reference vector."""
    Xt = PSeAAC().fit_transform(_frame(SEQ))
    np.testing.assert_allclose(Xt.to_numpy()[0], solution, atol=1e-3)


@pytest.mark.parametrize(
    "prop_indices,group_props,custom_groups,width",
    [
        ([0, 1, 2, 3, 4, 5], 2, None, 150),
        (None, None, [[0, 1], [2, 3], [4, 5], [6, 7]], 200),
    ],
)
def test_configurations(prop_indices, group_props, custom_groups, width):
    """Output width is (20 + lambda_val) times the number of property groups."""
    pse = PSeAAC(
        prop_indices=prop_indices, group_props=group_props, custom_groups=custom_groups
    )
    assert pse.fit_transform(_frame(SEQ)).shape == (1, width)


def test_group_props_and_custom_groups_conflict():
    """group_props and custom_groups cannot both be given."""
    with pytest.raises(ValueError, match="not both"):
        PSeAAC(group_props=3, custom_groups=[[0, 1, 2]]).fit_transform(_frame(SEQ))


def test_empty_custom_groups():
    """An empty custom_groups list raises instead of falling back to the default groups."""  # noqa: E501
    with pytest.raises(ValueError, match="at least one group"):
        PSeAAC(custom_groups=[]).fit_transform(_frame(SEQ))


def test_default_grouping_needs_multiple_of_three():
    """Without group_props, the selected property count must be divisible by 3."""
    with pytest.raises(ValueError, match="divisible by 3"):
        PSeAAC(prop_indices=[0, 1, 2, 3]).fit_transform(_frame(SEQ))


def test_indivisible_group_props():
    """group_props must divide the number of selected properties."""
    with pytest.raises(ValueError, match="divisible by group_props"):
        PSeAAC(prop_indices=[0, 1, 2, 3, 4], group_props=3).fit_transform(_frame(SEQ))


def test_case_insensitive():
    """Lowercase input gives the same vector as uppercase, without warnings."""
    seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lower = PSeAAC().fit_transform(_frame(seq.lower()))
    upper = PSeAAC().fit_transform(_frame(seq))
    pd.testing.assert_frame_equal(lower, upper)
