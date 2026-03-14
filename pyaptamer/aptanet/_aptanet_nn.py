__author__ = "satvshr"
__all__ = ["AptaNetMLP"]

import torch.nn as nn


def aptanet_layer(input_dim, output_dim, dropout, lazy=False):
    """
    Create a single AptaNet layer composed of a linear transformation,
    ReLU activation, and standard Dropout.
    """
    linear = nn.LazyLinear(output_dim) if lazy else nn.Linear(input_dim, output_dim)
    return nn.Sequential(
        linear,
        nn.ReLU(), # Use nn.SELU() here if you switch back to AlphaDropout
        nn.Dropout(dropout), # Changed from AlphaDropout to match ReLU
    )


class AptaNetMLP(nn.Module):
    """
    A fully-connected (vanilla) multi-layer perceptron (MLP).
    ... [Keep your original docstring here] ...
    """

    def __init__(
        self,
        input_dim=None,
        hidden_dim=128,
        n_hidden=7,
        dropout=0.3,
        output_dim=1,
        use_lazy=True,
    ):
        super().__init__()

        # Crash prevention: Check if user disabled lazy loading but forgot input_dim
        if not use_lazy and input_dim is None:
            raise ValueError("input_dim must be provided if use_lazy is False.")

        first_lazy = use_lazy and (input_dim is None)
        layers = []

        if n_hidden > 0:
            # First hidden layer
            layers.append(aptanet_layer(input_dim, hidden_dim, dropout, lazy=first_lazy))
            # Remaining hidden layers (n_hidden - 1)
            for _ in range(n_hidden - 1):
                layers.append(aptanet_layer(hidden_dim, hidden_dim, dropout, lazy=False))

        # Output layer
        layers.append(nn.Linear(hidden_dim if n_hidden > 0 else input_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Perform a forward pass through the network.
        ... [Keep your original docstring here] ...
        """
        return self.model(x)
