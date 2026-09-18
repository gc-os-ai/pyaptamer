"""Test the AptaDiff denoiser and diffusion wrapper."""

__author__ = ["aditi-dsi"]

import pytest
import torch
import torch.nn as nn

from pyaptamer.aptadiff._model import (
    AptaDiffDenoiser,
    AptaDiffDiffusion,
    _index_to_log_onehot,
    _log_onehot_to_index,
)

BATCH_SIZE = 2
NUM_CLASSES = 4
SEQ_LEN = 8
ENC_EMBED_SIZE = 16
TIMESTEPS = 4


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a batch of token indices and latent conditioning vectors."""
    x = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, SEQ_LEN))
    z = torch.randn(BATCH_SIZE, ENC_EMBED_SIZE)

    return x, z


@pytest.fixture
def denoiser_kwargs() -> dict:
    """Return keyword arguments for a small AptaDiffDenoiser."""
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
def denoiser(denoiser_kwargs: dict) -> AptaDiffDenoiser:
    """Build an AptaDiffDenoiser from denoiser_kwargs."""
    return AptaDiffDenoiser(**denoiser_kwargs)


class TestAptaDiffHelpers:
    """Tests for the log-one-hot encoding helpers in _model.py."""

    @pytest.mark.parametrize("batch_size, seq_len", [(1, 4), (2, 8), (3, 16)])
    def test_index_onehot_roundtrip(self, batch_size: int, seq_len: int) -> None:
        """Check _index_to_log_onehot and _log_onehot_to_index are inverses."""
        x = torch.randint(0, NUM_CLASSES, (batch_size, seq_len))

        log_onehot = _index_to_log_onehot(x, NUM_CLASSES)
        recovered_indices = _log_onehot_to_index(log_onehot)

        assert log_onehot.shape == (batch_size, NUM_CLASSES, seq_len), (
            f"Expected ({batch_size}, {NUM_CLASSES}, {seq_len}), "
            f"got {tuple(log_onehot.shape)}."
        )
        assert torch.equal(recovered_indices, x), (
            "Round-trip through log-one-hot did not recover the original indices."
        )

    @pytest.mark.parametrize("bad_index", [-1, NUM_CLASSES])
    def test_index_to_log_onehot_rejects_out_of_range(self, bad_index: int) -> None:
        """Check out-of-range class indices raise a ValueError."""
        x = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
        x[0, 0] = bad_index

        with pytest.raises(ValueError, match="class indices in"):
            _index_to_log_onehot(x, NUM_CLASSES)


class TestAptaDiffDenoiser:
    """Test AptaDiffDenoiser."""

    @torch.no_grad()
    def test_denoiser_forward_shape(
        self, denoiser: AptaDiffDenoiser, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check AptaDiffDenoiser produces (batch, num_classes, seq_len) logits."""
        x, z = batch
        t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,))
        out = denoiser(x, t, z)

        assert out.shape == (BATCH_SIZE, NUM_CLASSES, SEQ_LEN)

    @torch.no_grad()
    def test_denoiser_output_is_zero_at_init(
        self, denoiser: AptaDiffDenoiser, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check the zero-initialized scale makes every logit zero."""
        x, z = batch
        t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,))
        out = denoiser(x, t, z)

        assert not out.any()


class FullVLBRecorder(AptaDiffDiffusion):
    """Record whether compute_full_vlb is called."""

    ran_full_vlb = False

    def compute_full_vlb(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Set ran_full_vlb, then return the full VLB."""
        self.ran_full_vlb = True
        return super().compute_full_vlb(x, z)


class TestAptaDiffDiffusion:
    """Test AptaDiffDiffusion."""

    @pytest.fixture
    def diffusion(self, denoiser: AptaDiffDenoiser) -> AptaDiffDiffusion:
        """Build an AptaDiffDiffusion around the denoiser fixture."""
        return AptaDiffDiffusion(
            denoise_fn=denoiser, num_classes=NUM_CLASSES, num_timesteps=TIMESTEPS
        )

    def test_predict_start_rejects_bad_denoiser_shape(
        self, denoiser_kwargs: dict, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check a denoiser whose output_dim is not num_classes raises a ValueError."""
        x, z = batch
        denoiser_kwargs = {**denoiser_kwargs, "output_dim": NUM_CLASSES + 1}
        diffusion = AptaDiffDiffusion(
            denoise_fn=AptaDiffDenoiser(**denoiser_kwargs),
            num_classes=NUM_CLASSES,
            num_timesteps=TIMESTEPS,
        )
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        t = torch.zeros(BATCH_SIZE, dtype=torch.long)

        with pytest.raises(ValueError, match="denoise_fn must return logits"):
            diffusion.predict_start(log_x0, t, z)

    def test_diffusion_construction_buffers(self, diffusion: AptaDiffDiffusion) -> None:
        """Check schedule and statistics buffers are registered with correct shapes."""
        buffers = dict(diffusion.named_buffers(recurse=False))

        assert set(buffers) == {"log_alpha", "log_alphabar", "Lt_history", "Lt_count"}
        assert all(buffer.shape == (TIMESTEPS,) for buffer in buffers.values())

    @pytest.mark.parametrize(
        "option", [{"loss_type": "vb"}, {"parametrization": "eps"}]
    )
    def test_diffusion_invalid_options(
        self, denoiser: AptaDiffDenoiser, option: dict
    ) -> None:
        """Check an unsupported loss_type or parametrization raises a ValueError."""
        with pytest.raises(ValueError, match="must be"):
            AptaDiffDiffusion(denoise_fn=denoiser, **option)

    def test_q_sample_rejects_out_of_range_timestep(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check q_sample raises a ValueError for a timestep past the schedule."""
        x, _ = batch
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        t = torch.full((BATCH_SIZE,), TIMESTEPS)

        with pytest.raises(ValueError, match="timesteps in"):
            diffusion.q_sample(log_x0, t)

    def test_predict_reverse_step_rejects_out_of_range_timestep(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check predict_reverse_step raises a ValueError for a negative timestep."""
        x, z = batch
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        t = torch.full((BATCH_SIZE,), -1)

        with pytest.raises(ValueError, match="timesteps in"):
            diffusion.predict_reverse_step(log_x0, t, z)

    def test_log_prob_rejects_mismatched_batch(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check log_prob raises a ValueError when x and z batch sizes differ."""
        x, z = batch

        with pytest.raises(ValueError, match="same batch size"):
            diffusion.log_prob(x, z[:1])

    def test_sample_time_rejects_unknown_method(
        self, diffusion: AptaDiffDiffusion
    ) -> None:
        """Check sample_time raises a ValueError for an unknown sampling method."""
        with pytest.raises(ValueError, match="Unknown sample time method"):
            diffusion.sample_time(BATCH_SIZE, torch.device("cpu"), "xyz")

    @torch.no_grad()
    def test_q_sample_is_valid_log_one_hot(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check q_sample returns one class with probability 1 at every position."""
        x, _ = batch
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        t = torch.full((BATCH_SIZE,), TIMESTEPS - 1)

        log_xt = diffusion.q_sample(log_x0, t)

        assert log_xt.shape == (BATCH_SIZE, NUM_CLASSES, SEQ_LEN)
        assert torch.all((log_xt == 0).sum(dim=1) == 1)

    def test_sample_time_importance_favors_high_loss_timesteps(
        self, diffusion: AptaDiffDiffusion
    ) -> None:
        """Check a larger recorded loss never gets a smaller sampling probability."""
        n_draws = 64
        diffusion.Lt_count.fill_(11)
        diffusion.Lt_history.copy_(torch.arange(1, TIMESTEPS + 1, dtype=torch.float32))

        sampled_timesteps, sampled_probs = diffusion.sample_time(
            n_draws, torch.device("cpu"), method="importance"
        )

        assert sampled_timesteps.shape == sampled_probs.shape == (n_draws,)
        assert sampled_timesteps.dtype == torch.long
        by_timestep = torch.argsort(sampled_timesteps)
        assert torch.all(torch.diff(sampled_probs[by_timestep]) >= 0)

    def test_log_prob_updates_statistics_only_in_train_mode(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check log_prob updates the Lt buffers in train mode but not in eval mode."""
        x, z = batch

        diffusion.eval()
        diffusion.log_prob(x, z)

        assert not diffusion.Lt_count.any()
        assert not diffusion.Lt_history.any()

        diffusion.train()
        diffusion.log_prob(x, z)

        assert diffusion.Lt_count.sum() == BATCH_SIZE
        assert diffusion.Lt_history.any()

    @pytest.mark.parametrize("training", [True, False])
    def test_log_prob_uses_full_vlb_only_in_train_mode(
        self,
        denoiser: AptaDiffDenoiser,
        batch: tuple[torch.Tensor, torch.Tensor],
        training: bool,
    ) -> None:
        """Check vb_all uses compute_full_vlb in train mode and not in eval mode."""
        x, z = batch
        diffusion = FullVLBRecorder(
            denoise_fn=denoiser,
            num_classes=NUM_CLASSES,
            num_timesteps=TIMESTEPS,
            loss_type="vb_all",
        )
        diffusion.train(training)

        diffusion.log_prob(x, z)

        assert diffusion.ran_full_vlb is training

    @pytest.mark.parametrize(
        "option", [{}, {"loss_type": "vb_all"}, {"parametrization": "direct"}]
    )
    def test_log_prob_is_finite_and_differentiable(
        self,
        denoiser: AptaDiffDenoiser,
        batch: tuple[torch.Tensor, torch.Tensor],
        option: dict,
    ) -> None:
        """Check log_prob returns a finite loss and finite denoiser gradients."""
        x, z = batch
        nn.init.ones_(denoiser.scale)
        diffusion = AptaDiffDiffusion(
            denoise_fn=denoiser,
            num_classes=NUM_CLASSES,
            num_timesteps=TIMESTEPS,
            **option,
        )
        diffusion.train()

        log_prob = diffusion.log_prob(x, z)
        log_prob.sum().backward()

        assert log_prob.shape == (BATCH_SIZE,)
        assert torch.isfinite(log_prob).all()
        for param in denoiser.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_log_prob_with_alternative_num_classes(self, denoiser_kwargs: dict) -> None:
        """Check log_prob supports a num_classes other than four."""
        num_classes = 6
        denoiser_kwargs = {
            **denoiser_kwargs,
            "input_dim": num_classes,
            "output_dim": num_classes,
        }
        diffusion = AptaDiffDiffusion(
            denoise_fn=AptaDiffDenoiser(**denoiser_kwargs),
            num_classes=num_classes,
            num_timesteps=TIMESTEPS,
        )
        diffusion.train()

        x = torch.randint(0, num_classes, (BATCH_SIZE, SEQ_LEN))
        z = torch.randn(BATCH_SIZE, ENC_EMBED_SIZE)

        loss = diffusion.log_prob(x, z)

        assert loss.shape == (BATCH_SIZE,)
        assert torch.isfinite(loss).all()

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_log_prob_finite_under_autocast(
        self,
        diffusion: AptaDiffDiffusion,
        denoiser: AptaDiffDenoiser,
        batch: tuple[torch.Tensor, torch.Tensor],
        dtype: torch.dtype,
    ) -> None:
        """Check loss and gradients stay finite under CPU autocast."""
        x, z = batch

        nn.init.ones_(denoiser.scale)
        diffusion.train()

        with torch.autocast(device_type="cpu", dtype=dtype):
            loss = diffusion.log_prob(x, z)

        assert torch.isfinite(loss).all()
        loss.sum().backward()
        for param in denoiser.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
