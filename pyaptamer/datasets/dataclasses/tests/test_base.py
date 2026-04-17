"""Tests for BaseAptamerDataset.

The base is intentionally minimal: tag plumbing + an abstract `load()`.
Anything more (input coercion, scitype-specific shapes) lives in concrete
subclasses (APIDataset, MaskedDataset).
"""

import pytest

from pyaptamer.datasets.dataclasses._base import BaseAptamerDataset


def test_base_class_default_tags():
    tags = BaseAptamerDataset.get_class_tags()
    assert tags["object_type"] == "dataset"
    # scitype + X_inner_mtype are subclass-set; base defaults are None.
    assert tags["scitype"] is None
    assert tags["X_inner_mtype"] is None
    assert tags["has_y"] is True


def test_base_class_inherits_from_baseobject():
    from skbase.base import BaseObject

    assert issubclass(BaseAptamerDataset, BaseObject)


def test_base_load_raises_not_implemented():
    """Bare BaseAptamerDataset has no canonical form; load() must raise."""
    ds = BaseAptamerDataset()
    with pytest.raises(NotImplementedError, match="must implement load"):
        ds.load()


def test_base_init_takes_no_data_args():
    """Base no longer accepts x_apta/x_prot — those moved to APIDataset."""
    with pytest.raises(TypeError):
        BaseAptamerDataset(x_apta=["A"], x_prot=["M"])
