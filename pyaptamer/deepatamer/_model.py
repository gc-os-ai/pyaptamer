__author__ = "satvshr"
__all__ = ["DeepAptamer"]

import torch
import torch.nn as nn


class DeepAptamer(nn.Module):
    def __init__(self, optimizer=None):
        super().__init__()

        # Sequence branch (B, 35, 4)
        self.seq_conv = nn.Conv1d(in_channels=4, out_channels=12, kernel_size=1)
        self.seq_fc = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=-1),
        )

        # Shape branch (B, 126, 1)
        self.shape_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=100)
        self.shape_pool = nn.MaxPool1d(kernel_size=20)
        self.shape_fc = nn.Sequential(nn.Linear(1, 4), nn.ReLU())

        # Rest of the model
        self.bilstm = nn.LSTM(
            input_size=4,
            hidden_size=100,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.dropout = nn.Dropout(0.1)

        self.attn = nn.MultiheadAttention(embed_dim=200, num_heads=1, batch_first=True)

        self.head = nn.Linear(200, 2)

        self.optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=0.001)

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
