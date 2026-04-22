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
    """GRU cell similar to Keras's reset_after=True.

    Parameters
    ----------
    input_size : int
        Number of features in the input sequence layer `x`.
    hidden_size : int
        Number of features in the hidden state `h`.
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
        x : torch.Tensor
            Input feature tensor at the current graph step.
        h : torch.Tensor
            Previous recurrent hidden state tensor.

        Returns
        -------
        torch.Tensor
            The updated hidden state tensor.
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
    """Message passing convolution step for the graph neural network.

    Parameters
    ----------
    filters : int, optional
        Number of filters/channels in the hidden representations, default is 64.
    multiply : str or None, optional
        Determines bilinear processing behavior during message aggregation.
    bn_layer : bool, optional
        Whether to apply batch normalization.
    gru_layer : bool, optional
        Whether to use the KerasGRUCell during propagation.
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
        x : torch.Tensor
            Matrix of node features for the current input batch.
        pairs_prev : torch.Tensor
            Adjacency index pairs pointing from previous nodes in the sequence.
        pairs_next : torch.Tensor
            Adjacency index pairs pointing from subsequent nodes in the sequence.

        Returns
        -------
        torch.Tensor
            Aggregated and processed output node representations.
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
    """Averages output features based on the target projection dimension.

    Parameters
    ----------
    target_features : int, optional
        Number of target structural dimensions to project over, default is 1.
    filter_size : int, optional
        Dimensionality of the initial input feature space, default is 64.
    """

    def __init__(self, target_features=1, filter_size=64):
        super().__init__()
        self.target_features = target_features if target_features != 0 else 1
        self.pad_amount = filter_size % self.target_features
        self.group_size = (filter_size + self.pad_amount) // self.target_features

    def forward(self, x):
        """Pad and calculate local feature means over grouped channels.

        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor derived from CNN or GRU layers.

        Returns
        -------
        torch.Tensor
            Vector of dimension-averaged outputs appropriate for structural mapping.
        """
        if self.pad_amount > 0:
            x = F.pad(x, (0, self.pad_amount))
        x = x.reshape(-1, self.target_features, self.group_size)
        return x.mean(dim=-1).reshape(-1)


class DNAModel(nn.Module):
    """Graph neural network architecture for predicting DNA shape features.

    Parameters
    ----------
    input_features : int, optional
        Number of dimensions in the initial base encoding arrays, default is 4.
    filter_size : int, optional
        Number of latent channels for message passing layers, default is 64.
    mp_layers : int, optional
        Total number of stacked message passing layers (depth), default is 7.
    mp_steps : int, optional
        Propagation steps executed sequentially per layer (width), default is 1.
    base_features : int, optional
        Number of targeted mathematical structural predictions, default is 1.
    constraints : bool, optional
        If True, preserves intermediate layer outputs to supply combined residual
        feedback.
    selflayer : bool, optional
        If True, preserves the initial isolated single-node convolution directly.
    multiply : str or None, optional
        Specifies if explicit linear expansion factors are applied inside convolution.
    bn_layer : bool, optional
        If True, applies batch normalization directly across graph hidden states.
    gru_layer : bool, optional
        If True, employs gated recurrent processing within node propagation rounds.
    dropout_rate : float, optional
        Fraction of aggregated units to probabilistically zero out, default is 0.0.
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
        """Pass input DNA matrices through the graph network.

        Parameters
        ----------
        x : torch.Tensor
            Encoded feature mapping inputs representing initial bases or k-mers.
        pairs_prev : torch.Tensor
            Forward adjacency connections establishing graph edge directions.
        pairs_next : torch.Tensor
            Backward adjacency connections generating recurrent bidirectional flows.

        Returns
        -------
        torch.Tensor
            Final DNA shape score prediction matrix over sequence length.
        """
        # Conv1d (batch, channels, length)
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
