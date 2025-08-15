import torch
import torch.nn as nn


class DeepAptamer(nn.Module):
    def __init__(self, dropout=0.1, lstm_hidden=100):
        super().__init__()

        # Sequence branch (input: B x 35 x 4)
        self.seq_conv = nn.Conv1d(in_channels=4, out_channels=12, kernel_size=1)
        self.seq_fc = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 4), nn.Softmax(dim=-1)
        )

        # Shape branch (input: B x 126 x 1)
        self.shape_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=100)
        self.shape_fc = nn.Sequential(
            nn.MaxPool1d(kernel_size=20, stride=20), nn.Linear(1, 4), nn.ReLU()
        )

        # BiLSTM over concatenated features (36 x 4)
        self.bi1 = nn.LSTM(
            input_size=4, hidden_size=100, batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.bi2 = nn.LSTM(
            input_size=200,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * lstm_hidden, num_heads=1, batch_first=True
        )
        # Final classification head
        self.head = nn.Linear(2 * lstm_hidden, 2)  # logits

    def forward(self, x_seq, x_shape):
        # Sequence branch
        s = x_seq.permute(0, 2, 1)  # (B, 4, 35)
        s = self.seq_conv(s)  # (B, 12, 35)
        s = s.permute(0, 2, 1)  # (B, 35, 12)
        s = self.seq_fc(s)  # (B, 35, 4)

        # Shape branch
        h = x_shape.permute(0, 2, 1)  # (B, 1, 126)
        h = self.shape_conv(h)  # (B, 1, 27)
        h = h.permute(0, 2, 1)  # (B, 1, 1)
        h = self.shape_fc(h)  # (B, 1, 4)

        # Concatenate along time axis
        x = torch.cat([s, h], dim=1)  # (B, 36, 4)

        # BiLSTM stack
        x, _ = self.bi1(x)
        x = self.dropout1(x)

        x, (h_n, _) = self.bi2(x)
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*lstm_hidden)
        x = self.dropout2(x)

        return self.head(x)
