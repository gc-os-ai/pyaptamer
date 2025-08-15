import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepAptamer(nn.Module):
    def __init__(self):
        super().__init__()

        # Sequence branch (B, 35, 4) -> (B, 35, 4)
        self.seq_conv = nn.Conv1d(in_channels=4, out_channels=12, kernel_size=1)
        self.seq_fc = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 4), nn.Softmax(dim=-1)
        )

        # Shape branch (B, 126, 1) -> (B, 1, 4)
        self.shape_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=100)
        self.shape_fc = nn.Sequential(nn.Linear(1, 4), nn.ReLU())

        # BiLSTM stack (matching TF's 100 units per direction)
        self.bi1 = nn.LSTM(
            input_size=4, hidden_size=100, batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(0.1)

        self.bi2 = nn.LSTM(
            input_size=200, hidden_size=100, batch_first=True, bidirectional=True
        )
        self.dropout2 = nn.Dropout(0.1)

        # Attention (200 features from bi2)
        self.attn = nn.MultiheadAttention(embed_dim=200, num_heads=1, batch_first=True)

        # Classification head
        self.head = nn.Linear(200, 2)

    def forward(self, x_seq, x_shape):
        # ----- sequence branch
        s = x_seq.permute(0, 2, 1)  # (B, 4, 35)
        s = self.seq_conv(s)  # (B, 12, 35)
        s = s.permute(0, 2, 1)  # (B, 35, 12)
        s = self.seq_fc(s)  # (B, 35, 4)

        # ----- shape branch
        h = x_shape.permute(0, 2, 1)  # (B, 1, 126)
        h = self.shape_conv(h)  # (B, 1, 27)
        h = F.max_pool1d(h, kernel_size=20, stride=20)  # (B, 1, 1)
        h = h.transpose(1, 2)  # ensure last dim is features
        h = self.shape_fc(h)  # (B, 1, 4)

        # ----- concat along time axis: 35 + 1 = 36
        x = torch.cat([s, h], dim=1)  # (B, 36, 4)

        # ----- BiLSTM stack
        x, _ = self.bi1(x)  # (B, 36, 200)
        x = self.dropout1(x)
        x, _ = self.bi2(x)  # (B, 36, 200)
        x = self.dropout2(x)

        # ----- attention
        x, _ = self.attn(x, x, x)  # (B, 36, 200)

        # ----- classification
        logits = self.head(x)  # (B, 36, 2)
        return logits
