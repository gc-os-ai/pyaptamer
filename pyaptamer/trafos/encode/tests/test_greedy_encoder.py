import inspect

import pandas as pd

from pyaptamer.trafos.encode import GreedyEncoder


def test_get_test_params_is_classmethod_and_callable():
    descriptor = inspect.getattr_static(GreedyEncoder, "get_test_params")
    assert isinstance(descriptor, classmethod)

    params = GreedyEncoder.get_test_params()
    assert isinstance(params, list)
    assert len(params) >= 1
    assert all(isinstance(p, dict) for p in params)


def test_greedy_encoder_can_construct_from_test_params():
    for params in GreedyEncoder.get_test_params():
        enc = GreedyEncoder(**params)
        X = pd.DataFrame([["ACGU"], ["A"]])
        Xt = enc.fit_transform(X)
        assert Xt.shape[0] == X.shape[0]

