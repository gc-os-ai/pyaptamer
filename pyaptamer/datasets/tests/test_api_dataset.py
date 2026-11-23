"""Test suite for the APIDataset class."""

__author__ = ["sahilphad07-sudo"]

import numpy as np
import pytest

from pyaptamer.datasets.dataclasses._api import APIDataset

PROT_WORDS = {"P": 1, "1": 2, "2": 3, "3": 4}


def _make(apta, prot, y, split="test"):
    return APIDataset(
        x_apta=np.array(apta),
        x_prot=np.array(prot),
        y=np.array(y),
        apta_max_len=6,
        prot_max_len=4,
        prot_words=PROT_WORDS,
        split=split,
    )


def test_short_aptamer_keeps_samples_aligned():
    """A too-short aptamer is dropped together with its protein and label."""
    with pytest.warns(UserWarning, match="Dropped 1 sample"):
        ds = _make(
            ["ACGUACGU", "AC", "GGGUUUAA"],
            ["P1", "P2", "P3"],
            ["positive", "negative", "positive"],
        )

    assert len(ds) == 2
    assert len(ds.x_apta) == len(ds.x_prot) == len(ds.y) == 2
    # the surviving second sample must carry P3 and its own label, not P2's
    assert ds.x_prot[1].tolist() == [1, 4, 0, 0]
    assert ds.y[1].item() == 1


def test_short_aptamer_alignment_train_split():
    """Alignment also holds once reversed aptamers are appended in training."""
    with pytest.warns(UserWarning, match="Dropped 2 sample"):
        ds = _make(["ACGUACGU", "AC"], ["P1", "P2"], ["positive", "negative"], "train")

    assert len(ds.x_apta) == len(ds.x_prot) == len(ds.y) == 2
    assert ds.y.tolist() == [1, 1]


def test_no_warning_when_all_aptamers_valid(recwarn):
    """Nothing is dropped, so no warning is emitted."""
    ds = _make(["ACGUACGU", "GGGUUUAA"], ["P1", "P3"], ["positive", "negative"])

    assert len(ds) == 2
    assert not [w for w in recwarn if "Dropped" in str(w.message)]


def test_all_aptamers_too_short_raises():
    """If no aptamer can be encoded, fail with a clear error."""
    with pytest.raises(ValueError, match="long enough to form a triplet"):
        _make(["AC", "G"], ["P1", "P2"], ["positive", "negative"])
