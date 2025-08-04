__author__ = "satvshr"
__all__ = ["pipe"]
__required__ = ["python>=3.9,<3.12"]

import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from skorch import NeuralNetBinaryClassifier

from pyaptamer.aptanet.aptanet_nn import AptaNetMLP
from pyaptamer.utils._aptanet_utils import pairs_to_features

net = NeuralNetBinaryClassifier(
    module=AptaNetMLP,
    module__input_dim=128,
    module__hidden_dim=64,
    module__n_hidden=5,
    module__dropout=0.3,
    module__output_dim=1,
    module__use_lazy=False,
    criterion=nn.BCEWithLogitsLoss,
    max_epochs=20,
    lr=0.01,
)

feature_transformer = FunctionTransformer(
    func=pairs_to_features,
    validate=False,
    # Optional arguments for pairs_to_features
    # example: kw_args={'k': 4, 'pseaac_kwargs': {'lambda_value': 30}}
    kw_args={},
)

# Direct feature selection using tree-based model
feature_selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=300, max_depth=9, random_state=0),
    threshold="mean",
)

pipe = Pipeline(
    [
        ("features", feature_transformer),
        ("select", feature_selector),
        ("clf", net),
    ]
)
