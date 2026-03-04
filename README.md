# pyaptamer - AI for aptamer discovery

### sponsored by ecoSPECS

The python library for easy aptamer design.

|  | **[Tutorials](https://github.com/gc-os-ai/pyaptamer/tree/main/examples)** · **[Issue Tracker](https://github.com/gc-os-ai/pyaptamer/issues)** · **[Project Board](https://github.com/orgs/gc-os-ai/projects/1)** |
|---|---|
| **Open&#160;Source** | [![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/gc-os-ai/pyaptamer/blob/main/LICENSE) [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) | |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.gg/7uKdHfdcJG) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/german-center-for-open-source-ai/) |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/gc-os-ai/pyaptamer/release.yml?logo=github)](https://github.com/sktime/pytorch-forecasting/actions/workflows/pypi_release.yml) |
| **Code** | [![!pypi](https://img.shields.io/pypi/v/pytorch-forecasting?color=orange)](https://pypi.org/project/pyaptamer/) [![!python-versions](https://img.shields.io/pypi/pyversions/pyaptamer)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  |

## 🌟 Features

- ✅ aptamer design and optimization algorithms
- ✅ feature extraction from proteins and compounds
- ✅ compatible with `pdb` and `biopython`
- ✅ `scikit-learn`-like API - standardized and composable
- 🛠️ Easily extendable with plugins
- 📦 Minimal dependencies

NOTE: the package is in early development and not 100% feature complete - contributions appreciated!

---

## 🛠️ Usage

Below is a minimal example showing how to instantiate, train and save
pretrained weights for the AptaTrans model.  This is the same procedure you
would follow when re-training the network from scratch before uploading the
resulting checkpoint to the GC.OS HuggingFace repository.

```python
import torch
from torch.utils.data import DataLoader

from pyaptamer.aptatrans import (
    AptaTrans,
    AptaTransLightning,
    EncoderPredictorConfig,
)

# --- build a very small model for demonstration ---
apta_embedding = EncoderPredictorConfig(num_embeddings=4, target_dim=2, max_len=8)
prot_embedding = EncoderPredictorConfig(num_embeddings=4, target_dim=2, max_len=8)
model = AptaTrans(apta_embedding, prot_embedding, in_dim=4, n_encoder_layers=1, n_heads=1, conv_layers=[1,1,1])

# wrap in Lightning for training convenience
lightning_model = AptaTransLightning(model)

# dummy dataset (replace with real aptamer/protein sequences)
x_apta = torch.randint(0, 4, (16, 8))
x_prot = torch.randint(0, 4, (16, 8))
y = torch.randint(0, 2, (16, 1)).float()
loader = DataLoader(list(zip(x_apta, x_prot, y)), batch_size=4, shuffle=True)

trainer = torch.manual_seed(42)  # deterministic initialization for example
from lightning import Trainer
trainer = Trainer(max_epochs=1)
trainer.fit(lightning_model, loader)

# after training you can save a checkpoint to disk
model.save_pretrained("./weights/pretrained.pt")

# later you (or others) can load the weights again
new_model = AptaTrans(apta_embedding, prot_embedding, in_dim=4, n_encoder_layers=1, n_heads=1, conv_layers=[1,1,1])
new_model.load_pretrained_weights("./weights/pretrained.pt")
```

---

## ⚡ Installation

### Stable version from pip

NOTE: the package is not released yet on pypi - install from repo instead (unstable version)

```bash
pip install pyaptamer
```

### Latest unstable version

```bash
# Clone the repository
git clone https://github.com/gc-os-ai/pyaptamer.git

# Install dependencies
pip install -e .  # latest version
# or editable developer install
pip install -e ".[dev]"
```

---

## 🤝 Contributing

Contributions are welcome! 🎉

How to start: [find a good first issue](https://github.com/gc-os-ai/pyaptamer/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)

and/or join the [discord](https://discord.gg/7uKdHfdcJG) and ping the developers,
you can also ask for longer projects here.

Contributions and participation are subject to the GC.OS Code of Conduct.

---

## 🗺️ Roadmap

* more complete set of aptamer design and protein feature algorithms
* wider support for `cif` and/or `biopandas`
* integration of first-principles simulation tools
* Community feedback integration - suggest features on the [issue tracker!](https://github.com/gc-os-ai/pyaptamer/issues)

---

#### Team

The package is maintained in collaboration between [ecoSPECS](https://ecospecs.de/en/) and the [German Center for Open Source AI](https://gcos.ai/).

* German Center for Open Source AI
    * Franz Kiraly ([@fkiraly](https://www.github.com/fkiraly)) - primary point of contact (package)
    * Simon Blanke ([@simonblanke](https://www.github.com/simonblanke))
* ecoSPECS
    * Dennis Kubiczek ([@KubiczekD](https://www.github.com/KubiczekD)) - primary point of contact (domain/aptamers)
    * Fatih Yolcu ([@fat1hy0](https://www.github.com/fat1hy0))
    * Jakob Birke ([@jabirke](https://www.github.com/jabirke))
    * Kerstin Moser ([@KerstinMoser](https://www.github.com/KerstinMoser))
* European Summer of Code contributors 2025
    * Matteo Pinna ([@nennomp](https://www.github.com/nennomp))
    * Satvik Mishra ([@satvshr](https://www.github.com/satvshr))
    * Siddharth ([@siddharth7113](https://www.github.com/siddharth7113))
