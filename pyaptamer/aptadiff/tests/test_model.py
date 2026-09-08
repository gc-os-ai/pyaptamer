"""Test suite for AptaDiff's denoiser and diffusion wrapper."""

__author__ = ["aditi-dsi"]

import math

import pytest
import torch
import torch.nn as nn

from pyaptamer.aptadiff import _model
from pyaptamer.aptadiff._functional import log_add_exp
from pyaptamer.aptadiff._model import (
    AptaDiffDenoiser,
    AptaDiffDiffusion,
    Rezero,
    _index_to_log_onehot,
    _log_onehot_to_index,
)

BATCH_SIZE = 2
NUM_CLASSES = 4
SEQ_LEN = 8
ENC_EMBED_SIZE = 16
TIMESTEPS = 50


@pytest.fixture
def denoiser_kwargs():
    """Provides a set of AptaDiffDenoiser kwargs for testing purpose."""
    return {
        "enc_embed_size": ENC_EMBED_SIZE,
        "input_dim": NUM_CLASSES,
        "output_dim": NUM_CLASSES,
        "dim": 32,
        "depth": 1,
        "n_blocks": 1,
        "max_seq_len": SEQ_LEN,
        "num_timesteps": TIMESTEPS,
        "heads": 2,
        "local_attn_window_size": SEQ_LEN,
    }


@pytest.fixture
def denoiser(denoiser_kwargs):
    """Builds a AptaDiffDenoiser for use as the diffusion model's backbone."""
    return AptaDiffDenoiser(**denoiser_kwargs)


@pytest.fixture
def diffusion(denoiser):
    """Builds an AptaDiffDiffusion wrapping the denoiser fixture."""
    return AptaDiffDiffusion(
        denoise_fn=denoiser, num_classes=NUM_CLASSES, num_timesteps=TIMESTEPS
    )


@pytest.fixture
def batch():
    """Provides a small batch of token indices and latent conditioning vectors."""
    x = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, SEQ_LEN))
    z = torch.randn(BATCH_SIZE, ENC_EMBED_SIZE)

    return x, z


@pytest.mark.parametrize("batch_size, seq_len", [(1, 4), (2, 8), (3, 16)])
def test_index_onehot_roundtrip(batch_size, seq_len):
    """Verify _index_to_log_onehot and _log_onehot_to_index are inverses."""
    x = torch.randint(0, NUM_CLASSES, (batch_size, seq_len))

    log_onehot = _index_to_log_onehot(x, NUM_CLASSES)
    recovered = _log_onehot_to_index(log_onehot)

    assert log_onehot.shape == (batch_size, NUM_CLASSES, seq_len)
    assert torch.equal(recovered, x)


@pytest.mark.parametrize("bad_index", [-1, NUM_CLASSES])
def test_index_to_log_onehot_rejects_out_of_range(bad_index):
    """Verify out-of-range class indices raise a ValueError."""
    x = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    x[0, 0] = bad_index

    with pytest.raises(ValueError, match="class indices in"):
        _index_to_log_onehot(x, NUM_CLASSES)


def test_rezero_forward():
    """Verify Rezero scales its input by a learnable alpha initialized at zero."""
    rezero = Rezero()
    x = torch.randn(BATCH_SIZE, NUM_CLASSES, SEQ_LEN)

    out = rezero(x)

    assert torch.equal(out, torch.zeros_like(x))
    assert rezero.alpha.requires_grad


def test_denoiser_forward_shape(denoiser, batch):
    """Verify AptaDiffDenoiser produces (batch, num_classes, seq_len) logits."""
    x, z = batch
    t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,))

    out = denoiser(x, t, z)

    assert out.shape == (BATCH_SIZE, NUM_CLASSES, SEQ_LEN)


def test_predict_start_is_uniform_at_init(diffusion, batch):
    """Verify the zero-initialized Rezero gate makes predict_start uniform."""
    x, z = batch
    log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
    t = torch.zeros(BATCH_SIZE, dtype=torch.long)

    log_pred = diffusion.predict_start(log_x0, t, z)

    expected = torch.full_like(log_pred, -math.log(NUM_CLASSES))
    assert torch.allclose(log_pred, expected, atol=1e-6)


def test_predict_start_rejects_bad_denoiser_shape(batch):
    """Verify a denoiser returning the wrong logit shape raises a ValueError."""

    class WrongShapeDenoiser(nn.Module):
        def forward(self, x, t, z):
            return torch.randn(x.size(0), NUM_CLASSES + 1, x.size(1))

    x, z = batch
    diffusion = AptaDiffDiffusion(
        denoise_fn=WrongShapeDenoiser(),
        num_classes=NUM_CLASSES,
        num_timesteps=TIMESTEPS,
    )
    log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
    t = torch.zeros(BATCH_SIZE, dtype=torch.long)

    with pytest.raises(ValueError, match="denoise_fn must return logits"):
        diffusion.predict_start(log_x0, t, z)


def test_diffusion_construction_buffers(diffusion):
    """Verify schedule and statistics buffers are registered with correct shapes."""
    registered = dict(diffusion.named_buffers())
    state = diffusion.state_dict()

    for buffer_name in (
        "log_alpha",
        "log_1m_alpha",
        "log_alphabar",
        "log_1m_alphabar",
        "Lt_history",
        "Lt_count",
    ):
        assert buffer_name in registered
        assert buffer_name in state
        assert registered[buffer_name].shape == (TIMESTEPS,)


def test_diffusion_construction_schedule_sanity(diffusion):
    """Verify the noise schedule is a valid complementary log-probability pair."""
    zeros = torch.zeros(TIMESTEPS)

    assert torch.allclose(
        log_add_exp(diffusion.log_alpha, diffusion.log_1m_alpha), zeros, atol=1e-5
    )
    assert torch.allclose(
        log_add_exp(diffusion.log_alphabar, diffusion.log_1m_alphabar),
        zeros,
        atol=1e-5,
    )


@pytest.mark.parametrize("loss_type", ["invalid", "vb", ""])
def test_diffusion_invalid_loss_type(denoiser, loss_type):
    """Verify an unsupported loss_type raises a ValueError."""
    with pytest.raises(ValueError, match="loss_type must be"):
        AptaDiffDiffusion(denoise_fn=denoiser, loss_type=loss_type)


@pytest.mark.parametrize("parametrization", ["invalid", "eps", ""])
def test_diffusion_invalid_parametrization(denoiser, parametrization):
    """Verify an unsupported parametrization raises a ValueError."""
    with pytest.raises(ValueError, match="parametrization must be"):
        AptaDiffDiffusion(denoise_fn=denoiser, parametrization=parametrization)


@pytest.mark.parametrize(
    "invalid_call, match",
    [
        (
            lambda d, x, z, log_x0: d.q_sample(
                log_x0, torch.full((BATCH_SIZE,), TIMESTEPS)
            ),
            "timesteps in",
        ),
        (
            lambda d, x, z, log_x0: d.predict_reverse_step(
                log_x0, torch.full((BATCH_SIZE,), -1), z
            ),
            "timesteps in",
        ),
        (lambda d, x, z, log_x0: d.log_prob(x, z[:1]), "same batch size"),
        (
            lambda d, x, z, log_x0: d.sample_time(
                BATCH_SIZE, torch.device("cpu"), "bogus"
            ),
            "Unknown sample time method",
        ),
    ],
)
def test_rejects_invalid_arguments(diffusion, batch, invalid_call, match):
    """Verify guards on timestep range, batch alignment, and sampling method."""
    x, z = batch
    log_x0 = _index_to_log_onehot(x, NUM_CLASSES)

    with pytest.raises(ValueError, match=match):
        invalid_call(diffusion, x, z, log_x0)


def test_q_sample_is_valid_log_one_hot(diffusion, batch):
    """Verify q_sample returns a log-one-hot sample, not a distribution."""
    x, _ = batch
    log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
    t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,))

    log_sample = diffusion.q_sample(log_x0, t)
    prob_sample = torch.exp(log_sample)

    expected_ones = torch.ones(BATCH_SIZE, SEQ_LEN)
    assert torch.allclose(torch.sum(prob_sample, dim=1), expected_ones, atol=1e-5)
    assert torch.allclose(torch.amax(prob_sample, dim=1), expected_ones, atol=1e-5)


def test_sample_time_importance_branch(diffusion):
    """Verify importance sampling draws from the sqrt(Lt_history) statistics."""
    diffusion.Lt_count.fill_(11)
    diffusion.Lt_history.copy_(torch.arange(1, TIMESTEPS + 1, dtype=torch.float32))

    sampled_timesteps, sampled_probs = diffusion.sample_time(
        BATCH_SIZE, torch.device("cpu"), method="importance"
    )

    scores = torch.sqrt(diffusion.Lt_history + 1e-10) + 0.0001
    scores[0] = scores[1]
    expected_probs = scores / scores.sum()

    assert sampled_timesteps.shape == (BATCH_SIZE,)
    assert sampled_timesteps.dtype == torch.long
    assert torch.all((sampled_timesteps >= 0) & (sampled_timesteps < TIMESTEPS))
    assert torch.allclose(sampled_probs, expected_probs[sampled_timesteps], atol=1e-5)


def test_sample_time_decoder_term_substitution(diffusion):
    """Verify the t=0 decoder term's score is replaced by the t=1 score."""
    history = torch.ones(TIMESTEPS)
    history[0] = 1e12

    diffusion.Lt_count.fill_(11)
    diffusion.Lt_history.copy_(history)

    _, sampled_probs = diffusion.sample_time(
        BATCH_SIZE, torch.device("cpu"), method="importance"
    )

    uniform_probs = torch.full_like(sampled_probs, 1.0 / TIMESTEPS)
    assert torch.allclose(sampled_probs, uniform_probs, atol=1e-6)


def test_log_prob_train_mode_updates_history(diffusion, batch):
    """Verify log_prob in train mode returns a finite loss and mutates Lt buffers."""
    x, z = batch
    diffusion.train()

    history_before = diffusion.Lt_history.clone()

    loss = diffusion.log_prob(x, z)

    assert loss.shape == (BATCH_SIZE,)
    assert torch.isfinite(loss).all()
    assert not torch.equal(diffusion.Lt_history, history_before)
    assert diffusion.Lt_count.sum() == BATCH_SIZE


def test_log_prob_eval_mode_leaves_history_unchanged(diffusion, batch):
    """Verify log_prob in eval mode returns a finite loss without mutating buffers."""
    x, z = batch
    diffusion.eval()

    history_before = diffusion.Lt_history.clone()
    count_before = diffusion.Lt_count.clone()

    loss = diffusion.log_prob(x, z)

    assert loss.shape == (BATCH_SIZE,)
    assert torch.isfinite(loss).all()
    assert torch.equal(diffusion.Lt_history, history_before)
    assert torch.equal(diffusion.Lt_count, count_before)


@pytest.mark.parametrize("loss_type", ["vb_stochastic", "vb_all"])
def test_log_prob_finite_for_all_loss_types(denoiser_kwargs, batch, loss_type):
    """Verify both loss_type modes produce a finite training loss."""
    x, z = batch
    small_timesteps = 5
    denoiser_kwargs["num_timesteps"] = small_timesteps
    small_denoiser = AptaDiffDenoiser(**denoiser_kwargs)
    diffusion = AptaDiffDiffusion(
        denoise_fn=small_denoiser,
        num_classes=NUM_CLASSES,
        num_timesteps=small_timesteps,
        loss_type=loss_type,
    )
    diffusion.train()

    loss = diffusion.log_prob(x, z)

    assert torch.isfinite(loss).all()


def test_compute_full_vlb_accumulates_all_timesteps(
    monkeypatch, denoiser_kwargs, batch
):
    """Verify compute_full_vlb sums every timestep's term plus the prior."""
    x, z = batch
    small_timesteps = 4
    denoiser_kwargs["num_timesteps"] = small_timesteps
    diffusion = AptaDiffDiffusion(
        denoise_fn=AptaDiffDenoiser(**denoiser_kwargs),
        num_classes=NUM_CLASSES,
        num_timesteps=small_timesteps,
    )

    monkeypatch.setattr(
        _model, "compute_vlb_loss", lambda **kwargs: torch.ones(BATCH_SIZE)
    )

    total = diffusion.compute_full_vlb(x, z)

    log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
    expected = small_timesteps + diffusion.kl_prior(log_x0)
    assert torch.allclose(total, expected, atol=1e-5)


def test_log_prob_backward_populates_denoiser_gradients(diffusion, denoiser, batch):
    """Verify backward() on the training loss populates finite denoiser gradients."""
    x, z = batch
    diffusion.train()

    loss = diffusion.log_prob(x, z)
    loss.sum().backward()

    grad_count = 0
    for param in denoiser.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        grad_count += 1

    assert grad_count > 0


def test_vb_all_backward_populates_gradients(denoiser_kwargs, batch):
    """Verify gradients reach the denoiser through the checkpointed vb_all path."""
    x, z = batch
    small_timesteps = 4
    denoiser_kwargs["num_timesteps"] = small_timesteps
    small_denoiser = AptaDiffDenoiser(**denoiser_kwargs)
    diffusion = AptaDiffDiffusion(
        denoise_fn=small_denoiser,
        num_classes=NUM_CLASSES,
        num_timesteps=small_timesteps,
        loss_type="vb_all",
    )
    diffusion.train()

    loss = diffusion.log_prob(x, z)
    loss.sum().backward()

    for param in small_denoiser.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_log_prob_finite_under_cpu_bf16_autocast(diffusion, denoiser, batch):
    """Verify loss and gradients stay finite under CPU bfloat16 autocast."""
    x, z = batch
    diffusion.train()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = diffusion.log_prob(x, z)

    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for param in denoiser.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
