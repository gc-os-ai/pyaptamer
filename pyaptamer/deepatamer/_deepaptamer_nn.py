__author__ = "satvshr"
__all__ = ["DeepAptamerNN"]

import torch
import torch.nn as nn


class DeepAptamerNN(nn.Module):
    """
    DeepAptamer neural network model for aptamer–protein interaction prediction.

    This architecture integrates:

    - A sequence branch using convolutional and fully-connected layers to
      process one-hot encoded aptamer sequences.
    - A structural (DNA shape) branch using convolution + pooling + dense layers to
        extract shape features using `deepDNAshape` from the aptamer sequence.
    - A BiLSTM for capturing sequential dependencies.
    - Multi-head self-attention for contextual feature refinement.
    - A final classification head for binary binding prediction.

    Parameters
    ----------
    seq_conv_in : int, optional, default=4
        Number of input channels for the sequence convolution branch. Typically 4
        for one-hot DNA encoding.

    seq_conv_out : int, optional, default=12
        Number of output channels (filters) for the sequence convolution.

    seq_conv_kernel_size : int, optional, default=1
        Kernel size for the sequence convolution.

    seq_pool_kernel_size : int, optional, default=1
        Kernel size for max-pooling after sequence convolution.

    seq_pool_stride : int, optional, default=1
        Stride for max-pooling after sequence convolution.

    seq_linear_hidden_dim : int, optional, default=32
        Hidden layer size for fully connected layers in the sequence branch.

    seq_conv_linear_out : int, optional, default=4
        Dimensionality of the output feature vector from the sequence branch.

    shape_conv_kernel_size : int, optional, default=100
        Kernel size for convolution in the shape branch.

    shape_pool_kernel_size : int, optional, default=20
        Kernel size for pooling in the shape branch.

    shape_pool_stride : int, optional, default=20
        Stride for pooling in the shape branch.

    bilstm_hidden_size : int, optional, default=100
        Number of hidden units in each LSTM direction.

    bilstm_num_layers : int, optional, default=2
        Number of BiLSTM layers.

    dropout : float, optional, default=0.1
        Dropout probability applied after the BiLSTM.

    optimizer : torch.optim.Optimizer or None, optional, default=None
        Optimizer for training. If None, defaults to Adam with lr=0.001.

    Attributes
    ----------
    seq_conv : nn.Conv1d
        1D convolution layer for sequence branch.

    seq_fc : nn.Sequential
        Fully connected projection for sequence features.

    shape_conv_pool : nn.Sequential
        Convolution + pooling for DNA shape features.

    shape_fc : nn.Sequential
        Fully connected projection for shape features.

    bilstm : nn.LSTM
        Bidirectional LSTM for sequential modeling.

    dropout : nn.Dropout
        Dropout layer applied after BiLSTM.

    attn : nn.MultiheadAttention
        Attention layer for contextual refinement.

    head : nn.Linear
        Final classification layer (logits for 2 classes).
    """

    def __init__(
        self,
        seq_conv_in=4,
        seq_conv_out=12,
        seq_conv_kernel_size=1,
        seq_pool_kernel_size=1,
        seq_pool_stride=1,
        seq_linear_hidden_dim=32,
        # Defines the size of the 1st dimension(time/seq) used for branch concatenation
        seq_conv_linear_out=4,
        shape_conv_kernel_size=100,
        shape_pool_kernel_size=20,
        shape_pool_stride=20,
        bilstm_hidden_size=100,
        bilstm_num_layers=2,
        dropout=0.1,
        optimizer=None,
    ):
        super().__init__()
        self.seq_conv_in = seq_conv_in
        self.seq_conv_out = seq_conv_out
        self.seq_conv_kernel_size = seq_conv_kernel_size
        self.seq_pool_kernel_size = seq_pool_kernel_size
        self.seq_pool_stride = seq_pool_stride
        self.seq_linear_hidden_dim = seq_linear_hidden_dim
        self.seq_conv_linear_out = seq_conv_linear_out
        self.shape_conv_kernel_size = shape_conv_kernel_size
        self.shape_pool_kernel_size = shape_pool_kernel_size
        self.shape_pool_stride = shape_pool_stride
        self.bilstm_hidden_size = bilstm_hidden_size
        self.bilstm_num_layers = bilstm_num_layers
        self.dropout_val = dropout
        self.optimizer = optimizer

        # Sequence branch (B, seq_len, 4)
        self.seq_conv = nn.Conv1d(
            in_channels=self.seq_conv_in,
            out_channels=self.seq_conv_out,
            kernel_size=self.seq_conv_kernel_size,
        )
        self.seq_fc = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=self.seq_pool_kernel_size, stride=self.seq_pool_stride
            ),
            nn.Linear(self.seq_conv_out, self.seq_linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.seq_linear_hidden_dim, self.seq_conv_linear_out),
            nn.Softmax(dim=-1),
        )

        # Shape branch (B, 1, 126/138)
        self.shape_conv_pool = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=1, kernel_size=self.shape_conv_kernel_size
            ),
            nn.MaxPool1d(
                kernel_size=self.shape_pool_kernel_size, stride=self.shape_pool_stride
            ),
        )
        self.shape_fc = nn.Sequential(nn.Linear(1, self.seq_conv_linear_out), nn.ReLU())

        # Rest of the model
        self.bilstm = nn.LSTM(
            input_size=self.seq_conv_linear_out,
            hidden_size=self.bilstm_hidden_size,
            num_layers=self.bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_val,
        )
        self.dropout = nn.Dropout(self.dropout_val)

        self.attn = nn.MultiheadAttention(
            embed_dim=2 * self.bilstm_hidden_size, num_heads=1, batch_first=True
        )

        self.head = nn.Linear(2 * bilstm_hidden_size, 2)

        self.optimizer = self.optimizer or torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x_ohe, x_shape):
        """
        Forward pass of the DeepAptamerNN model.

        Parameters
        ----------
        x_ohe : torch.Tensor
            One-hot encoded aptamer sequence tensor of shape (batch_size, seq_len, 4).
            Example: (B, seq_len, 4) for batch size B and sequence length seq_len.

        x_shape : torch.Tensor
            DNA shape feature tensor of shape (batch_size, 1, shape_len).
            Example: (B, 1, 126) for batch size B and shape feature length 126.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, 2), representing the predicted
            class scores for binary classification.
        """
        out_ohe = x_ohe.permute(0, 2, 1)
        out_ohe = self.seq_conv(out_ohe)
        out_ohe = out_ohe.permute(0, 2, 1)
        out_ohe = self.seq_fc(out_ohe)

        out_shape = self.shape_conv_pool(x_shape)
        out_shape = out_shape.transpose(1, 2)
        out_shape = self.shape_fc(out_shape)

        x = torch.cat([out_ohe, out_shape], dim=1)

        x, _ = self.bilstm(x)

        x = x[:, -1, :]
        x = self.dropout(x)

        x = x.unsqueeze(1)
        x, _ = self.attn(x, x, x)

        x = x.squeeze(1)

        return self.head(x)
