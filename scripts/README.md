# scripts

Maintainer utilities that are not part of the installed `pyaptamer` package.

## `convert_deepdnashape.py`

Converts upstream [deepDNAshape](https://github.com/JinsenLi/deepDNAshape)
TensorFlow checkpoints into PyTorch `.pt` weight files used by
`pyaptamer.deepdnashape` (`DNAModel`).

deepDNAshape predicts DNA structural shape features (e.g. minor groove width
`MGW`, propeller twist `ProT`, `Roll`, helical twist `HelT`) from sequence
with a graph neural network: each base (or base-step) is a node, message
passing mixes neighbors, and a DualBias GRU updates node states. Each shape
feature has its own trained weights.

Upstream ships those weights as TensorFlow checkpoints
(`Feature.index` + `Feature.data-00000-of-00001`). This script maps the TF
variables onto the Torch `state_dict` layout expected by pyaptamer and writes
one `Feature.pt` per checkpoint.

### Requirements

- `numpy`
- `tensorflow` (to read checkpoints)
- `torch` (to write `.pt` files)
- optional: an installed `pyaptamer` checkout if you use `--validate`

### Usage

```bash
# Interactive prompts for input / output folders
python scripts/convert_deepdnashape.py

# Or pass paths explicitly
python scripts/convert_deepdnashape.py \
  --models-dir /path/to/tf_weights \
  --outdir converted_weights \
  --validate
```

The input folder should contain files like `MGW.index`,
`MGW.data-00000-of-00001`, and the same for other features. The script
discovers every `*.index` present and writes matching `.pt` files to the
output folder.
