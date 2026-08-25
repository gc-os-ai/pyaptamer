# Vendored third-party code

This folder contains code extracted and modified from three separate
`lucidrains` (Phil Wang) repositories, for internal use by
`pyaptamer.aptadiff.layers._transformer`. They are vendored to avoid
taking on three separate PyPI dependencies for a small amount of code
each. These files are private implementation details and are not part
of pyaptamer's public API.

## Contents and modifications

* `_axial_positional_embedding.py`
  Source: https://github.com/lucidrains/axial-positional-embedding
  - Removed the `AxialPositionalEmbeddingImage` wrapper class (image
    input, unused here) and its `einops` dependency.
  - `AxialPositionalEmbedding` itself is unmodified logic.

* `_local_attention.py`
  Source: https://github.com/lucidrains/local-attention
  - Removed causal attention, rotary positional embeddings, `shared_qk`,
    `autopad`, and custom-scale support. Only fixed-window, non-causal
    local attention is kept.
  - Removed the `mask` / `input_mask` / `attn_bias` forward arguments.
  - Replaced `einops.rearrange` with `torch.Tensor.view` to drop the
    `einops` dependency.

* `_linear_attention_transformer.py`
  Source: https://github.com/lucidrains/linear-attention-transformer
  - Removed causal attention, Linformer support, product-key-memory
    layers, reversible-network execution, axial folding, token
    shifting, rotary positional embeddings, and the standalone
    `LinearAttentionTransformerLM` language-model wrapper.
  - `SelfAttention` and `LinearAttentionTransformer` keep only the
    parameters this codebase uses; unsupported upstream parameters
    (e.g. `causal`, `reversible`, `dim_head`) are not accepted.

None of these changes alter the numerical behavior of the code paths
that are kept; they remove code paths unused by this codebase.

## License

All three source repositories are authored by Phil Wang (`lucidrains`)
and distributed under the MIT License. See [`LICENSE`](./LICENSE) for
the full text, which applies only to the files listed above. The rest
of pyaptamer is licensed under the terms in the repository root
`LICENSE` file.
