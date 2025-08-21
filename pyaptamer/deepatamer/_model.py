__author__ = "satvshr"
__all__ = ["DeepAptamerNN"]

import torch
import torch.nn as nn


class DeepAptamerNN(nn.Module):
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
        bilstm_dropout=0.1,
        post_bilstm_dropout=0.1,
        optimizer=None,
    ):
        super().__init__()
        self.seq_conv_in = seq_conv_in
        self.seq_conv_out = seq_conv_out
        self.seq_conv_kernel_size = seq_conv_kernel_size
        self.seq_linear_hidden_dim = seq_linear_hidden_dim
        self.seq_conv_linear_out = seq_conv_linear_out
        self.shape_conv_kernel_size = shape_conv_kernel_size
        self.shape_pool_kernel_size = shape_pool_kernel_size
        self.shape_pool_stride = shape_pool_stride
        self.bilstm_hidden_size = bilstm_hidden_size
        self.bilstm_num_layers = bilstm_num_layers
        self.bilstm_dropout = bilstm_dropout
        self.post_bilstm_dropout = post_bilstm_dropout
        self.optimizer = optimizer

        # Sequence branch (B, 35, 4)
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

        # Shape branch (B, 1, 126)
        self.shape_conv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=self.shape_conv_kernel_size
        )
        self.shape_pool = nn.MaxPool1d(
            kernel_size=self.shape_pool_kernel_size, stride=self.shape_pool_stride
        )
        self.shape_fc = nn.Sequential(nn.Linear(1, self.seq_conv_linear_out), nn.ReLU())

        # Rest of the model
        self.bilstm = nn.LSTM(
            input_size=self.seq_conv_linear_out,
            hidden_size=self.bilstm_hidden_size,
            num_layers=self.bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.bilstm_dropout,
        )
        self.dropout = nn.Dropout(self.post_bilstm_dropout)

        self.attn = nn.MultiheadAttention(
            embed_dim=2 * self.bilstm_hidden_size, num_heads=1, batch_first=True
        )

        self.head = nn.Linear(2 * bilstm_hidden_size, 2)

        self.optimizer = self.optimizer or torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x_ohe, x_shape):
        s = x_ohe.permute(0, 2, 1)
        s = self.seq_conv(s)
        s = s.permute(0, 2, 1)
        s = self.seq_fc(s)

        h = self.shape_conv(x_shape)
        h = self.shape_pool(h)
        h = h.transpose(1, 2)
        h = self.shape_fc(h)

        x = torch.cat([s, h], dim=1)

        x, _ = self.bilstm(x)

        x = x[:, -1, :]
        x = self.dropout(x)

        x = x.unsqueeze(1)
        x, _ = self.attn(x, x, x)

        x = x.squeeze(1)

        return self.head(x)

    def train_loop(self, X_ohe, X_shape, y, epochs=10, device="cpu"):
        self.to(device)
        criterion = nn.CrossEntropyLoss()

        X_ohe, X_shape, y = X_ohe.to(device), X_shape.to(device), y.to(device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            logits = self(X_ohe, X_shape)
            loss = criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
