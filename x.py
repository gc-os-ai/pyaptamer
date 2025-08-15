import torch
import torch.nn as nn
import torch.optim as optim

from pyaptamer.deepatamer.model import (
    DeepAptamer,
)  # Assuming this is the correct import path

# For now, I’ll assume DeepAptamer is defined above in the same script

# Create model
model = DeepAptamer()

# Dummy inputs
batch_size = 8
x_seq = torch.randn(batch_size, 35, 4)  # (B, 35, 4)
x_shape = torch.randn(batch_size, 126, 1)  # (B, 126, 1)

# Dummy binary labels for each time step (B, 36, 2)
y = torch.randint(0, 2, (batch_size, 36, 2)).float()

# Loss & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # small loop for demo
    model.train()

    optimizer.zero_grad()

    # Forward pass
    outputs = model(x_seq, x_shape)  # (B, 36, 2)

    # Compute loss
    loss = criterion(outputs, y)

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

print("Training complete!")
