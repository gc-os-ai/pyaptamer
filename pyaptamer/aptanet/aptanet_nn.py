import torch.nn as nn


def aptanet_layer(input_dim, output_dim, dropout, lazy=False):
    """Create a single AptaNet layer with AlphaDropout and ReLU activation."""
    linear = nn.LazyLinear(output_dim) if lazy else nn.Linear(input_dim, output_dim)
    return nn.Sequential(
        linear,
        nn.ReLU(),
        nn.AlphaDropout(dropout),
    )


class AptaNetMLP(nn.Module):
    """
    Vanilla MLP with optional lazy first layer.

    Parameters
    ----------
    input_dim : int or None
        If None and use_lazy=True, first layer is nn.LazyLinear.
    hidden_dim : int
    n_hidden : int
    dropout : float
    output_dim : int
    use_lazy : bool
        If True, ignore input_dim and build first layer lazily.
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

        first_lazy = use_lazy and (input_dim is None)

        layers = [aptanet_layer(input_dim, hidden_dim, dropout, lazy=first_lazy)]
        for _ in range(n_hidden):
            layers.append(aptanet_layer(hidden_dim, hidden_dim, dropout, lazy=False))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
