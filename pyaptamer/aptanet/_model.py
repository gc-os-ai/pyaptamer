__author__ = "satvshr"
__all__ = ["AptaNetMLP"]

import torch.nn as nn


def aptanet_layer(input_dim, output_dim, dropout, lazy=False):
    """
    Create a single AptaNet layer composed of a linear transformation,
    ReLU activation, and AlphaDropout.

    Parameters
    ----------
    input_dim : int
        Size of each input sample. Ignored if `lazy=True`.

    output_dim : int
        Size of each output sample (i.e., number of neurons in the layer).

    dropout : float
        Dropout probability for AlphaDropout. Must be between 0 and 1.

    lazy : bool, optional
        If True, use `nn.LazyLinear` instead of `nn.Linear`, allowing the input
        size to be inferred at runtime. Default is False.

    Returns
    -------
    nn.Sequential
        A sequential container with:
        - Linear or LazyLinear layer
        - ReLU activation
        - AlphaDropout layer
    """
    linear = nn.LazyLinear(output_dim) if lazy else nn.Linear(input_dim, output_dim)
    return nn.Sequential(
        linear,
        nn.ReLU(),
        nn.AlphaDropout(dropout),
    )


class AptaNetMLP(nn.Module):
    """
    A fully-connected (vanilla) multi-layer perceptron (MLP).

    This model supports lazy initialization for the first linear layer if the input
    dimension is unknown at instantiation time. Hidden layers use a customizable number
    of layers, hidden units, and dropout.

    Parameters
    ----------
    input_dim : int or None, optional
        Dimensionality of the input features. If None and `use_lazy=True`, the first
        layer is lazily initialized. Required if `use_lazy=False`. Default is None.

    hidden_dim : int, optional
        Number of units in each hidden layer. Default is 128.

    n_hidden : int, optional
        Number of hidden layers. Default is 7.

    dropout : float, optional
        Dropout rate applied after each hidden layer. Default is 0.3.

    output_dim : int, optional
        Dimensionality of the output layer. Typically 1 for binary classification or
        regression. Default is 1.

    use_lazy : bool, optional
        If True, the first layer is initialized lazily using `nn.LazyLinear`.
        Default is True.

    Attributes
    ----------
    model : nn.Sequential
        The sequential container of layers making up the MLP.
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
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim). If using lazy layers,
            the input shape will determine the first layer's dimensions at runtime.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim), containing logits.
        """
        return self.model(x)
