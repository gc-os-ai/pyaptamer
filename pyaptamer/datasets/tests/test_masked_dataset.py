import random

from pyaptamer.datasets.dataclasses._masked import MaskedDataset


def test_masked_tokens_follow_bert_distribution(monkeypatch):
    seq = list(range(1, 101))
    ds = MaskedDataset([seq], [seq], max_len=100, mask_idx=999, masked_rate=1.0)

    monkeypatch.setattr(random, "sample", lambda population, k: list(range(k)))
    rolls = iter([0.05] * 80 + [0.85] * 10 + [0.95] * 10)
    monkeypatch.setattr(random, "random", lambda: next(rolls))
    monkeypatch.setattr(random, "choice", lambda options: 777)

    x_masked, y_masked, x_orig, y_orig = ds[0]

    assert x_masked.tolist().count(999) == 80
    assert x_masked.tolist().count(777) == 10
    assert (
        sum(1 for idx, value in enumerate(x_masked.tolist()) if value == seq[idx]) == 10
    )
    assert y_masked.tolist() == seq
    assert x_orig.tolist() == seq
    assert y_orig.tolist() == seq
