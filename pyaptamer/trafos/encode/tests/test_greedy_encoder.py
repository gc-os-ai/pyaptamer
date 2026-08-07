import inspect

import pandas as pd
import pytest

from pyaptamer.trafos.encode import GreedyEncoder


def test_get_test_params_is_classmethod_and_callable():
    descriptor = inspect.getattr_static(GreedyEncoder, "get_test_params")
    assert isinstance(descriptor, classmethod)

    params_default = GreedyEncoder.get_test_params(parameter_set="default")
    params = GreedyEncoder.get_test_params()
    assert params == params_default
    assert isinstance(params, list)
    assert len(params) >= 1
    assert all(isinstance(p, dict) for p in params)


def test_get_test_params_unknown_parameter_set_raises():
    with pytest.raises(ValueError, match='parameter_set must be "default"'):
        GreedyEncoder.get_test_params(parameter_set="not_a_real_set")


def test_greedy_encoder_can_construct_from_test_params():
    for params in GreedyEncoder.get_test_params():
        enc = GreedyEncoder(**params)
        X = pd.DataFrame([["ACGU"], ["A"]])
        Xt = enc.fit_transform(X)
        assert Xt.shape[0] == X.shape[0]

