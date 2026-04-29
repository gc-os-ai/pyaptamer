"""Graph neural network model for DNA shape prediction."""

__author__ = ["prashantpandeygit"]
__all__ = ["DNAModel"]

import torch
import torch.nn as nn
import torch.nn.functional as F


def _segment_sum(data, segment_ids, num_segments):
    """Sum rows of data grouped by segment_ids

    Parameters
    ----------
    data : torch.Tensor, shape (E, C)
        Values to sum.
    segment_ids : torch.Tensor, shape (E,)
        Segment index for each row of data.
    num_segments : int
        Total number of output segments.

    Returns
    -------
    torch.Tensor, shape (num_segments, C)
    """
    result = torch.zeros(
        num_segments, data.shape[1], dtype=data.dtype, device=data.device
    )
    idx = segment_ids.unsqueeze(1).expand_as(data)
    result.scatter_add_(0, idx, data)
    return result


class KerasGRUCell(nn.Module):
    """Gated Recurrent Unit (GRU) cell.

    This implementation uses separate bias terms for the input and
    recurrent matrix multiplications. Unlike a standard single-bias
    GRU, the reset gate logic applies the recurrent bias *after* the
    matrix product is computed.

    Parameters
    ----------
    input_size : int
        Dimensionality of each input feature vector.
    hidden_size : int
        Dimensionality of the hidden state vector.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel = nn.Parameter(torch.empty(input_size, 3 * hidden_size))
        self.rec_kernel = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))
        self.bias = nn.Parameter(torch.empty(2, 3 * hidden_size))

    def forward(self, x, h):
        """Forward pass for Keras GRU cell.

        Parameters
        ----------
        x : torch.Tensor of shape (N, input_size)
            Input features for the current step.
        h : torch.Tensor of shape (N, hidden_size)
            Hidden state from the previous step.

        Returns
        -------
        torch.Tensor of shape (N, hidden_size)
            Updated hidden state.
        """
        matrix_x = x @ self.kernel + self.bias[0]
        x_z, x_r, x_h = torch.split(matrix_x, self.hidden_size, dim=-1)

        matrix_inner = h @ self.rec_kernel + self.bias[1]
        rec_z, rec_r, rec_h = torch.split(matrix_inner, self.hidden_size, dim=-1)

        z = torch.sigmoid(x_z + rec_z)
        r = torch.sigmoid(x_r + rec_r)

        h_candidate = torch.tanh(x_h + r * rec_h)
        return z * h + (1 - z) * h_candidate


class MessagePassingConv(nn.Module):
    """Graph convolution layer that aggregates features from adjacent nodes.

    Each position in the DNA sequence is treated as a node in a
    linear (chain) graph. This layer collects the feature vectors
    from each node's immediate predecessor and successor, combines
    them through learned weight matrices, and optionally refines
    the result with batch normalization and a `KerasGRUCell`.
    Stacking multiple layers allows information to propagate
    across longer sequence distances.

    Parameters
    ----------
    filters : int, optional
        Number of channels in the hidden node representations.
        Default is 64.
    multiply : str or None, optional
        Aggregation mode. "add" applies a second set of
        learned weights and an element-wise product with the
        current node state. None uses a simple residual
        (additive skip) connection. Default is None.
    bn_layer : bool, optional
        Whether to apply batch normalization.
    gru_layer : bool, optional
        If True, refine the aggregated features with a
        `KerasGRUCell`. If False, apply a sigmoid
        activation instead. Default is True.
    """

    def __init__(
        self,
        filters: int = 64,
        multiply: str | None = None,
        bn_layer: bool = True,
        gru_layer: bool = True,
    ):
        super().__init__()
        self.filters = filters
        self.multiply = multiply

        self.w_next = nn.Parameter(torch.empty(filters, filters))
        self.w_prev = nn.Parameter(torch.empty(filters, filters))
        self.b = nn.Parameter(torch.zeros(1, filters))
        nn.init.normal_(self.w_next)
        nn.init.normal_(self.w_prev)

        if multiply:
            if multiply == "add":
                self.w_next_all = nn.Parameter(torch.empty(filters, filters))
                self.w_prev_all = nn.Parameter(torch.empty(filters, filters))
                nn.init.normal_(self.w_next_all)
                nn.init.normal_(self.w_prev_all)
            self.b_all = nn.Parameter(torch.zeros(1, filters))

        # keras defaults eps=1e-3, momentum=0.99 (pytorch momentum = 1 - 0.99 = 0.01)
        self.bn = nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01) if bn_layer else None
        self.gru = KerasGRUCell(filters, filters) if gru_layer else None

    def forward(self, x, pairs_prev, pairs_next):
        """Execute one iteration of graph message passing.

        Parameters
        ----------
        x : torch.Tensor of shape (N, filters)
            Current node feature matrix, one row per sequence
            position.
        pairs_prev : torch.Tensor of shape (E, 2)
            Edge index pairs [target, source] for edges
            pointing from predecessor nodes.
        pairs_next : torch.Tensor of shape (E, 2)
            Edge index pairs [target, source] for edges
            pointing from successor nodes.

        Returns
        -------
        torch.Tensor of shape (N, filters)
            Updated node feature matrix.
        """
        num_nodes = x.shape[0]

        prev_x = x[pairs_prev[:, 1]]
        prev_sumx = _segment_sum(prev_x, pairs_prev[:, 0], num_nodes)

        next_x = x[pairs_next[:, 1]]
        next_sumx = _segment_sum(next_x, pairs_next[:, 0], num_nodes)

        aggre = next_sumx @ self.w_next + prev_sumx @ self.w_prev + self.b

        if self.multiply:
            if self.multiply == "add":
                aggre = (
                    aggre + next_sumx @ self.w_next_all + prev_sumx @ self.w_prev_all
                )
            aggre = aggre * x + self.b_all
        else:
            aggre = aggre + x

        if self.bn is not None:
            aggre = self.bn(F.relu(aggre))

        if self.gru is not None:
            x = self.gru(aggre, x)
        else:
            x = torch.sigmoid(aggre)

        return x


class AvgFeatures(nn.Module):
    """Dimensionality reduction layer that averages sub-features.

    Given a feature vector of size filter_size at each node,
    this layer divides the channels into target_features equal
    groups (zero-padding if filter_size is not evenly
    divisible), computes the mean within each group, and returns
    one scalar per target feature per node.

    Parameters
    ----------
    target_features : int, optional
        Number of output scalars per node. Each scalar is the
        mean of a group of channels. Default is 1.
    filter_size : int, optional
        Number of input channels to partition. Default is 64.
    """

    def __init__(self, target_features=1, filter_size=64):
        super().__init__()
        self.target_features = target_features if target_features != 0 else 1
        self.pad_amount = filter_size % self.target_features
        self.group_size = (filter_size + self.pad_amount) // self.target_features

    def forward(self, x):
        """Average channel groups into target output features.

        Parameters
        ----------
        x : torch.Tensor of shape (N, filter_size)
            Node feature matrix from a preceding convolution or
            GRU layer.

        Returns
        -------
        torch.Tensor of shape (N * target_features,)
            Flattened vector of per-node, per-feature averages.
        """
        if self.pad_amount > 0:
            x = F.pad(x, (0, self.pad_amount))
        x = x.reshape(-1, self.target_features, self.group_size)
        return x.mean(dim=-1).reshape(-1)


class DNAModel(nn.Module):
    """Graph neural network for predicting DNA shape features.

    The model treats each position in a DNA sequence as a node in a
    linear (chain) graph.  It first projects the one-hot encoded input
    through a 1-D convolution, then applies a configurable stack of
    `MessagePassingConv` layers where each node exchanges features
    with its neighbours through learned weight matrices and optional
    GRU-based updates. After each layer an `AvgFeatures` reduction
    predictions are stacked to form the final output.

    Original Implementation: https://github.com/JinsenLi/deepDNAshape

    Parameters
    ----------
    input_features : int, optional
        Number of input channels per node. 4 for single-base
        (intra-base pair) features, 16 for di-nucleotide
        (inter-base pair) features. Default is 4.
    filter_size : int, optional
        Number of hidden channels used throughout the
        convolution and GRU layers. Default is 64.
    mp_layers : int, optional
        Number of stacked `MessagePassingConv` layers.
        Default is 7.
    mp_steps : int, optional
        Number of consecutive forward passes through each
        `MessagePassingConv` layer before moving to the
        next. Default is 1.
    base_features : int, optional
        Number of output scalars per node produced by the
        `AvgFeatures` reduction. Default is 1.
    constraints : bool, optional
        If True, collect an `AvgFeatures` prediction after
        every `MessagePassingConv` layer and stack them
        along a new axis. Default is True.
    selflayer : bool, optional
        If True, collect an `AvgFeatures` prediction from
        the initial convolution output before any message
        passing. Default is True.
    multiply : str or None, optional
        Aggregation mode forwarded to each
        `MessagePassingConv`. Default is "add".
    bn_layer : bool, optional
        If True, enable batch normalization inside each
        `MessagePassingConv`. Default is True.
    gru_layer : bool, optional
        If True, enable GRU refinement inside each
        `MessagePassingConv`. Default is True.
    dropout_rate : float, optional
        Dropout probability applied to node features before
        the `AvgFeatures` reduction during training.
        Default is 0.0.
    """

    def __init__(
        self,
        input_features=4,
        filter_size=64,
        mp_layers=7,
        mp_steps=1,
        base_features=1,
        constraints=True,
        selflayer=True,
        multiply="add",
        bn_layer=True,
        gru_layer=True,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.steps = mp_steps
        self.constraints = constraints
        self.selflayer = selflayer

        self.input_conv = nn.Conv1d(input_features, filter_size, 1)

        self.mp = nn.ModuleList(
            [
                MessagePassingConv(
                    filters=filter_size,
                    multiply=multiply,
                    bn_layer=bn_layer,
                    gru_layer=gru_layer,
                )
                for _ in range(mp_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.avg_layer = AvgFeatures(base_features, filter_size)

    def _call_avg(self, x):
        if self.training:
            x = self.dropout(x)
        return self.avg_layer(x)

    def forward(self, x, pairs_prev, pairs_next):
        """Run the full forward pass of the graph neural network.

        Parameters
        ----------
        x : torch.Tensor of shape (N, input_features)
            One-hot (or di-nucleotide) encoded sequence, one
            row per node.
        pairs_prev : torch.Tensor of shape (E, 2)
            Predecessor edge indices produced by
            _build_graph.
        pairs_next : torch.Tensor of shape (E, 2)
            Successor edge indices produced by
            _build_graph.

        Returns
        -------
        torch.Tensor
            Final DNA shape score prediction matrix over sequence length.
        """
        x = x.unsqueeze(0).permute(0, 2, 1)
        x = self.input_conv(x)
        x = x.permute(0, 2, 1).squeeze(0)

        results = []
        if self.selflayer:
            results.append(self._call_avg(x))

        for layer in self.mp:
            for _ in range(self.steps):
                x = layer(x, pairs_prev, pairs_next)
            if self.constraints:
                results.append(self._call_avg(x))

        if self.constraints:
            return torch.stack(results, dim=1)
        return self._call_avg(x)
