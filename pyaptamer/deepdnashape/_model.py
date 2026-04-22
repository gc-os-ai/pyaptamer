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
    """Memory based processor that learns patterns in DNA sequences.

    Parameters
    ----------
    input_size : int
        The size of the incoming data signals for each DNA base.
    hidden_size : int
        The size of the internal memory state used to track patterns.
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
            The memory state accumulated from previous positions.

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
    """A layer that allows neighboring DNA bases to share information.

    Parameters
    ----------
    filters : int, optional
        The width of the information flow between bases, default 64.
    multiply : str or None, optional
        Internal math setting for combining different types of signals.
    bn_layer : bool, optional
        Whether to apply batch normalization.
    gru_layer : bool, optional
        Whether to use the Keras-style memory unit during propagation.
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
            The current list of features for each position in the sequence.
        pairs_prev : torch.Tensor
            Map of which positions are connected to their predecessors.
        pairs_next : torch.Tensor
            Map of which positions are connected to their successors.

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
    """Compresses internal data into simple physical measurements.

    Parameters
    ----------
    target_features : int, optional
        How many distinct physical values to output (e.g., 1 for Width).
    filter_size : int, optional
        The size of the complex data coming from the previous layers.
    """

    def __init__(self, target_features=1, filter_size=64):
        super().__init__()
        self.target_features = target_features if target_features != 0 else 1
        self.pad_amount = filter_size % self.target_features
        self.group_size = (filter_size + self.pad_amount) // self.target_features

    def forward(self, x):
        """Calculates the final physical average for each DNA position.

        Parameters
        ----------
        x : torch.Tensor
            The raw signals produced by the internal layers.

        Returns
        -------
        torch.Tensor
            The actual physical shape measurement for each part of the DNA.
        """
        if self.pad_amount > 0:
            x = F.pad(x, (0, self.pad_amount))
        x = x.reshape(-1, self.target_features, self.group_size)
        return x.mean(dim=-1).reshape(-1)


class DNAModel(nn.Module):
    """Graph neural network architecture for predicting DNA shape features.

    Original implementation: https://github.com/JinsenLi/deepDNAshape

    Parameters
    ----------
    input_features : int, optional
        Number of data signals per initial DNA base, default is 4.
    filter_size : int, optional
        Width of the internal processing channels, default is 64.
    mp_layers : int, optional
        Total number of stacked message passing layers (depth), default is 7.
    mp_steps : int, optional
        How many iterations of information sharing per layer, default is 2.
    base_features : int, optional
        Number of output physical properties to predict, default is 1.
    constraints : bool, optional
        If True, keeps track of intermediate work for more stability.
    selflayer : bool, optional
        If True, performs an initial scan before sharing info.
    multiply : str or None, optional
        Detailed math setting for how signals are combined.
    bn_layer : bool, optional
        If True, uses stabilization layers inside.
    gru_layer : bool, optional
        If True, uses memory based information processing.
    dropout_rate : float, optional
        Reduces overfitting by ignoring random signals during training.
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
            The initial encoded sequence of DNA bases.
        pairs_prev : torch.Tensor
            Connection map for information flowing one way.
        pairs_next : torch.Tensor
            Connection map for information flowing the other way.

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
