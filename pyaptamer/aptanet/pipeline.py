import torch
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from pyaptamer.aptanet import FeatureSelector, SkorchAptaNet
from pyaptamer.utils._aptanet_utils import pairs_to_features

net = SkorchAptaNet(
    module__hidden_dim=128,
    module__n_hidden=7,
    module__dropout=0.3,
    max_epochs=200,
    lr=1.4e-4,
    batch_size=310,
    optimizer=optim.RMSprop,
    device="cuda" if torch.cuda.is_available() else "cpu",
    threshold=0.5,
    verbose=1,
)

feature_transformer = FunctionTransformer(
    func=pairs_to_features,
    validate=False,
    # Optional arguments for pairs_to_features
    # example: kw_args={'k': 4, 'pseaac_kwargs': {'lambda_value': 30}}
    kw_args={},
)

pipe = Pipeline(
    [("features", feature_transformer), ("select", FeatureSelector()), ("clf", net)]
)
