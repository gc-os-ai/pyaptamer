import pytest
import torch

from pyaptamer.aptadiff.layers._transformer import AptaDiffTransformerEmbedding


@pytest.fixture
def default_transformer_kwargs():
    """Provides a standard set of hyperparameter kwargs for instantiation."""
    return {
        "enc_embed_size": 16,
        "input_dim": 4,
        "output_dim": 32,
        "dim": 64,
        "depth": 2,
        "n_blocks": 2,
        "max_seq_len": 128,
        "num_timesteps": 100,
        "heads": 4,
        "local_attn_window_size": 64,
        "attn_layer_dropout": 0.0,
    }


@pytest.mark.parametrize(
    "transformer_type, n_local_attn_heads",
    [
        ("native", 0),
        ("linear", 0),
        ("linear", 2),
    ],
)
def test_transformer_initialization(
    default_transformer_kwargs, transformer_type, n_local_attn_heads
):
    """Test successful initialization across specific backend and head configs."""
    default_transformer_kwargs["n_local_attn_heads"] = n_local_attn_heads

    model = AptaDiffTransformerEmbedding(
        **default_transformer_kwargs, transformer_type=transformer_type
    )

    assert model is not None
    assert model.emb_dim == 64
    assert len(model.transformer_blocks) == 2


def test_transformer_invalid_type(default_transformer_kwargs):
    """Test that an invalid transformer_type raises a ValueError."""
    with pytest.raises(
        ValueError, match="transformer_type must be 'linear' or 'native'"
    ):
        AptaDiffTransformerEmbedding(
            **default_transformer_kwargs, transformer_type="invalid"
        )


def test_transformer_native_warning_on_local_heads(default_transformer_kwargs):
    """Test that using local heads with native attention triggers a warning."""
    with pytest.warns(
        UserWarning, match="n_local_attn_heads is set to 4 but will be ignored"
    ):
        AptaDiffTransformerEmbedding(
            **default_transformer_kwargs,
            transformer_type="native",
            n_local_attn_heads=4,
        )


@pytest.mark.parametrize("max_seq_len, window_size", [(100, 30), (128, 50), (64, 10)])
def test_transformer_invalid_max_seq_len_window_ratio(
    default_transformer_kwargs, max_seq_len, window_size
):
    """Test that max_seq_len must be divisible by local_attn_window_size."""
    default_transformer_kwargs["max_seq_len"] = max_seq_len
    default_transformer_kwargs["local_attn_window_size"] = window_size

    with pytest.raises(
        ValueError, match="must be evenly divisible by local_attn_window_size"
    ):
        AptaDiffTransformerEmbedding(**default_transformer_kwargs)


@pytest.mark.parametrize(
    "transformer_type, n_local_attn_heads",
    [
        ("native", 0),
        ("linear", 0),
        ("linear", 2),
    ],
)
@pytest.mark.parametrize(
    "batch_size, seq_len, window_size", [(2, 128, 64), (1, 64, 64), (4, 128, 32)]
)
def test_transformer_forward_pass(
    default_transformer_kwargs,
    transformer_type,
    n_local_attn_heads,
    batch_size,
    seq_len,
    window_size,
):
    """Test the forward pass tensor shapes across different backends and batch sizes."""
    default_transformer_kwargs["max_seq_len"] = seq_len
    default_transformer_kwargs["local_attn_window_size"] = window_size
    default_transformer_kwargs["n_local_attn_heads"] = n_local_attn_heads

    model = AptaDiffTransformerEmbedding(
        **default_transformer_kwargs, transformer_type=transformer_type
    )

    vocab_size = default_transformer_kwargs["input_dim"]
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    t = torch.randint(0, 100, (batch_size,))
    z = torch.randn(batch_size, default_transformer_kwargs["enc_embed_size"])

    out = model(x, t, z)

    expected_shape = (batch_size, seq_len, default_transformer_kwargs["output_dim"])
    assert out.shape == expected_shape


def test_transformer_forward_invalid_seq_len_local_attention(
    default_transformer_kwargs,
):
    """Test that forward raises ValueError when seq_len not divisible by the window
    once local attention heads are active."""
    default_transformer_kwargs["n_local_attn_heads"] = 2
    model = AptaDiffTransformerEmbedding(
        **default_transformer_kwargs, transformer_type="linear"
    )

    vocab_size = default_transformer_kwargs["input_dim"]
    batch_size, bad_seq_len = 2, 100
    x = torch.randint(0, vocab_size, (batch_size, bad_seq_len))
    t = torch.randint(0, 100, (batch_size,))
    z = torch.randn(batch_size, default_transformer_kwargs["enc_embed_size"])

    with pytest.raises(ValueError, match="must be evenly divisible by window_size"):
        model(x, t, z)


def test_transformer_forward_exceeds_max_seq_len(default_transformer_kwargs):
    """Test that forward raises ValueError when seq_len exceeds max_seq_len."""
    model = AptaDiffTransformerEmbedding(**default_transformer_kwargs)

    vocab_size = default_transformer_kwargs["input_dim"]
    batch_size = 2
    long_seq_len = default_transformer_kwargs["max_seq_len"] + 10

    x = torch.randint(0, vocab_size, (batch_size, long_seq_len))
    t = torch.randint(0, 100, (batch_size,))
    z = torch.randn(batch_size, default_transformer_kwargs["enc_embed_size"])

    with pytest.raises(
        ValueError, match="must be less than the maximum sequence length"
    ):
        model(x, t, z)
