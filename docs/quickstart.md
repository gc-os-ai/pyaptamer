
---

# 📄 `docs/quickstart.md`

```md
# Quick Start

This guide shows a simple example of how to use PyAptamer.

## Example

```python
from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig
import torch

# Create embeddings
apta_embedding = EncoderPredictorConfig(128, 16, max_len=128)
prot_embedding = EncoderPredictorConfig(128, 16, max_len=128)

# Initialize model
model = AptaTrans(apta_embedding, prot_embedding, pretrained=False)

# Dummy input
x_apta = torch.randint(0, 16, (2, 10))
x_prot = torch.randint(0, 16, (2, 12))

# Forward pass
output = model(x_apta, x_prot)

print(output.shape)