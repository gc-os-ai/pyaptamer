import torch
import torch.nn as nn


class DeepAptamer(nn.Module):
    def __init__(self, dropout=0.1, lstm_hidden=100):
        super().__init__()

        # Sequence branch (input: B x 35 x 4) model1
        self.seq_conv = nn.Conv1d(in_channels=4, out_channels=12, kernel_size=1)
        self.seq_dense = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # no activation, passthrough features
            nn.ReLU(),
        )

        # Shape branch (input: B x 126 x 1) model2
        self.shape_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=100)
        self.shape_pool = nn.MaxPool1d(kernel_size=20, stride=20)  # 27 -> 1
        self.shape_dense = nn.Sequential(nn.Linear(1, 4), nn.ReLU())

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

        # Final classification head
        self.head = nn.Linear(2 * lstm_hidden, 2)  # logits

    def forward(self, x_seq, x_shape):
        # Sequence branch
        s = x_seq.permute(0, 2, 1)  # B x 4 x 35
        s = self.seq_conv(s)  # B x 12 x 35
        s = s.permute(0, 2, 1)  # B x 35 x 12
        s = self.seq_dense(s)  # B x 35 x 4

        # Shape branch
        h = x_shape.permute(0, 2, 1)  # B x 1 x 126
        h = self.shape_conv(h)  # B x 1 x 27
        h = self.shape_pool(h)  # B x 1 x 1
        h = h.permute(0, 2, 1)  # B x 1 x 1
        h = self.shape_dense(h)  # B x 1 x 4

        # Concatenate along sequence length axis: (35+1) = 36
        x = torch.cat([s, h], dim=1)  # B x 36 x 4

        # BiLSTM stack
        x, _ = self.bi1(x)  # B x 36 x 200
        x = self.dropout1(x)
        x, (hn, _) = self.bi2(x)  # hn: (2, B, hidden)
        x = torch.cat([hn[-2], hn[-1]], dim=1)  # B x (2*lstm_hidden)
        x = self.dropout2(x)

        logits = self.head(x)  # B x 2 (logits)
        return logits
