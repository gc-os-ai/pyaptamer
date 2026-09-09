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
SMALL_TIMESTEPS = 4


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Provides a fixed-size batch of token indices and latent conditioning vectors."""
    x = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, SEQ_LEN))
    z = torch.randn(BATCH_SIZE, ENC_EMBED_SIZE)

    return x, z


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
    """Tests for the Rezero() gate and the AptaDiffDenoiser() class."""

    def test_rezero_forward(self) -> None:
        """Check Rezero scales its input by a learnable alpha initialized at zero."""
        rezero = Rezero()
        x = torch.randn(BATCH_SIZE, NUM_CLASSES, SEQ_LEN)

        out = rezero(x)

        assert torch.equal(out, torch.zeros_like(x))
        assert rezero.alpha.requires_grad

    @pytest.mark.parametrize(
        "batch_size, seq_len, dim",
        [(2, 8, 32), (4, 16, 64), (1, 4, 16)],
    )
    @torch.no_grad()
    def test_denoiser_forward_shape(
        self, batch_size: int, seq_len: int, dim: int
    ) -> None:
        """Check AptaDiffDenoiser produces (batch, num_classes, seq_len) logits."""
        denoiser = AptaDiffDenoiser(
            enc_embed_size=ENC_EMBED_SIZE,
            input_dim=NUM_CLASSES,
            output_dim=NUM_CLASSES,
            dim=dim,
            depth=1,
            n_blocks=1,
            max_seq_len=seq_len,
            num_timesteps=TIMESTEPS,
            heads=2,
            local_attn_window_size=seq_len,
        )

        x = torch.randint(0, NUM_CLASSES, (batch_size, seq_len))
        t = torch.randint(0, TIMESTEPS, (batch_size,))
        z = torch.randn(batch_size, ENC_EMBED_SIZE)

        out = denoiser(x, t, z)

        assert out.shape == (batch_size, NUM_CLASSES, seq_len), (
            f"Expected ({batch_size}, {NUM_CLASSES}, {seq_len}), "
            f"got {tuple(out.shape)}."
        )


class TestAptaDiffDiffusion:
    """Tests for the AptaDiffDiffusion() class."""

    @pytest.fixture
    def denoiser_kwargs(self) -> dict:
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
    def denoiser(self, denoiser_kwargs: dict) -> AptaDiffDenoiser:
        """Builds an AptaDiffDenoiser instance for use as the model's backbone."""
        return AptaDiffDenoiser(**denoiser_kwargs)

    @pytest.fixture
    def diffusion(self, denoiser: AptaDiffDenoiser) -> AptaDiffDiffusion:
        """Builds an AptaDiffDiffusion instance wrapping the denoiser fixture."""
        return AptaDiffDiffusion(
            denoise_fn=denoiser, num_classes=NUM_CLASSES, num_timesteps=TIMESTEPS
        )

    @torch.no_grad()
    def test_predict_start_is_uniform_at_init(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check the zero-initialized Rezero gate makes predict_start uniform."""
        x, z = batch
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        t = torch.zeros(BATCH_SIZE, dtype=torch.long)

        log_pred = diffusion.predict_start(log_x0, t, z)

        expected = torch.full_like(log_pred, -math.log(NUM_CLASSES))
        assert torch.allclose(log_pred, expected, atol=1e-6)

    def test_predict_start_rejects_bad_denoiser_shape(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check a denoiser returning the wrong logit shape raises a ValueError."""

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

    def test_predict_reverse_step_direct_parametrization(
        self, denoiser: AptaDiffDenoiser, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check parametrization="direct" returns the denoiser prediction unchanged.

        Under "direct" the reverse step must bypass q_posterior entirely, so the
        output has to match predict_start exactly rather than the posterior it
        would produce under "x0".
        """
        x, z = batch
        diffusion = AptaDiffDiffusion(
            denoise_fn=denoiser,
            num_classes=NUM_CLASSES,
            num_timesteps=TIMESTEPS,
            parametrization="direct",
        )
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,))

        log_model_pred = diffusion.predict_reverse_step(log_x0, t, z)
        log_start_pred = diffusion.predict_start(log_x0, t, z)

        assert log_model_pred.shape == (BATCH_SIZE, NUM_CLASSES, SEQ_LEN)
        assert torch.allclose(log_model_pred, log_start_pred, atol=1e-6)

    @pytest.mark.parametrize("t_value", [0, TIMESTEPS - 1])
    def test_boundary_timesteps_are_accepted(
        self,
        diffusion: AptaDiffDiffusion,
        batch: tuple[torch.Tensor, torch.Tensor],
        t_value: int,
    ) -> None:
        """Check the first and last valid timesteps flow through the model path."""
        x, z = batch
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        t = torch.full((BATCH_SIZE,), t_value, dtype=torch.long)

        log_xt = diffusion.q_sample(log_x0, t)
        log_pred = diffusion.predict_reverse_step(log_xt, t, z)

        assert log_xt.shape == (BATCH_SIZE, NUM_CLASSES, SEQ_LEN)
        assert log_pred.shape == (BATCH_SIZE, NUM_CLASSES, SEQ_LEN)
        assert torch.isfinite(log_pred).all()

    def test_diffusion_construction_buffers(self, diffusion: AptaDiffDiffusion) -> None:
        """Check schedule and statistics buffers are registered with correct shapes."""
        registered = dict(diffusion.named_buffers())
        parameters = dict(diffusion.named_parameters())
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
            assert buffer_name not in parameters
            assert buffer_name in state
            assert registered[buffer_name].shape == (TIMESTEPS,)

    @torch.no_grad()
    def test_diffusion_construction_schedule_sanity(
        self, diffusion: AptaDiffDiffusion
    ) -> None:
        """Check the noise schedule is a valid complementary log-probability pair."""
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
    def test_diffusion_invalid_loss_type(
        self, denoiser: AptaDiffDenoiser, loss_type: str
    ) -> None:
        """Check an unsupported loss_type raises a ValueError."""
        with pytest.raises(ValueError, match="loss_type must be"):
            AptaDiffDiffusion(denoise_fn=denoiser, loss_type=loss_type)

    @pytest.mark.parametrize("parametrization", ["invalid", "eps", ""])
    def test_diffusion_invalid_parametrization(
        self, denoiser: AptaDiffDenoiser, parametrization: str
    ) -> None:
        """Check an unsupported parametrization raises a ValueError."""
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
                    BATCH_SIZE, torch.device("cpu"), "xyz"
                ),
                "Unknown sample time method",
            ),
        ],
    )
    def test_rejects_invalid_arguments(
        self,
        diffusion: AptaDiffDiffusion,
        batch: tuple[torch.Tensor, torch.Tensor],
        invalid_call,
        match: str,
    ) -> None:
        """Check guards on timestep range, batch alignment, and sampling method."""
        x, z = batch
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)

        with pytest.raises(ValueError, match=match):
            invalid_call(diffusion, x, z, log_x0)

    @torch.no_grad()
    def test_q_sample_is_valid_log_one_hot(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check q_sample returns a log-one-hot sample."""
        x, _ = batch
        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,))

        log_sample = diffusion.q_sample(log_x0, t)
        prob_sample = torch.exp(log_sample)

        expected_ones = torch.ones(BATCH_SIZE, SEQ_LEN)
        assert torch.allclose(torch.sum(prob_sample, dim=1), expected_ones, atol=1e-5)
        assert torch.allclose(torch.amax(prob_sample, dim=1), expected_ones, atol=1e-5)

    def test_sample_time_importance_returns_aligned_indices_and_probs(
        self, diffusion: AptaDiffDiffusion
    ) -> None:
        """Check importance sampling returns in-range indices aligned to their probs."""
        n_draws = 64
        diffusion.Lt_count.fill_(11)
        diffusion.Lt_history.copy_(torch.arange(1, TIMESTEPS + 1, dtype=torch.float32))

        sampled_timesteps, sampled_probs = diffusion.sample_time(
            n_draws, torch.device("cpu"), method="importance"
        )

        assert sampled_timesteps.shape == (n_draws,)
        assert sampled_probs.shape == (n_draws,)
        assert sampled_timesteps.dtype == torch.long
        assert torch.all((sampled_timesteps >= 0) & (sampled_timesteps < TIMESTEPS))
        assert torch.all((sampled_probs > 0.0) & (sampled_probs <= 1.0))

        for t_value in torch.unique(sampled_timesteps):
            probs_for_t = sampled_probs[sampled_timesteps == t_value]
            assert torch.allclose(probs_for_t, probs_for_t[0]), (
                f"Timestep {int(t_value)} was reported with differing probabilities."
            )

    def test_sample_time_importance_favors_high_loss_timesteps(
        self, diffusion: AptaDiffDiffusion
    ) -> None:
        """Check a larger recorded loss never gets a smaller sampling weight."""
        n_draws = 64
        diffusion.Lt_count.fill_(11)
        diffusion.Lt_history.copy_(torch.arange(1, TIMESTEPS + 1, dtype=torch.float32))

        sampled_timesteps, sampled_probs = diffusion.sample_time(
            n_draws, torch.device("cpu"), method="importance"
        )

        history = diffusion.Lt_history[sampled_timesteps]
        by_history = torch.argsort(history)

        assert torch.all(torch.diff(sampled_probs[by_history]) >= -1e-6)

    def test_sample_time_decoder_term_substitution(
        self, diffusion: AptaDiffDiffusion
    ) -> None:
        """Check the t=0 decoder term's score is replaced by the t=1 score."""
        history = torch.ones(TIMESTEPS)
        history[0] = 1e12

        diffusion.Lt_count.fill_(11)
        diffusion.Lt_history.copy_(history)

        _, sampled_probs = diffusion.sample_time(
            BATCH_SIZE, torch.device("cpu"), method="importance"
        )

        uniform_probs = torch.full_like(sampled_probs, 1.0 / TIMESTEPS)
        assert torch.allclose(sampled_probs, uniform_probs, atol=1e-6)

    def test_log_prob_train_mode_updates_importance_statistics(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check log_prob in train mode updates importance tracking buffers."""
        x, z = batch
        diffusion.train()

        history_before = diffusion.Lt_history.clone()

        diffusion.log_prob(x, z)

        assert diffusion.Lt_count.sum() == BATCH_SIZE
        assert not torch.equal(diffusion.Lt_history, history_before)

    def test_log_prob_eval_mode_leaves_statistics_unchanged(
        self, diffusion: AptaDiffDiffusion, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check log_prob in eval mode leaves the Lt buffers unchanged."""
        x, z = batch
        diffusion.eval()

        history_before = diffusion.Lt_history.clone()
        count_before = diffusion.Lt_count.clone()

        diffusion.log_prob(x, z)

        assert torch.equal(diffusion.Lt_history, history_before)
        assert torch.equal(diffusion.Lt_count, count_before)

    @pytest.mark.parametrize(
        "loss_type, training",
        [("vb_stochastic", True), ("vb_all", True), ("vb_stochastic", False)],
    )
    def test_log_prob_returns_finite_loss(
        self,
        denoiser_kwargs: dict,
        batch: tuple[torch.Tensor, torch.Tensor],
        loss_type: str,
        training: bool,
    ) -> None:
        """Check log_prob returns a finite loss in every mode and loss type."""
        x, z = batch
        denoiser_kwargs = {**denoiser_kwargs, "num_timesteps": SMALL_TIMESTEPS}
        small_denoiser = AptaDiffDenoiser(**denoiser_kwargs)
        diffusion = AptaDiffDiffusion(
            denoise_fn=small_denoiser,
            num_classes=NUM_CLASSES,
            num_timesteps=SMALL_TIMESTEPS,
            loss_type=loss_type,
        )
        diffusion.train(training)

        loss = diffusion.log_prob(x, z)

        assert loss.shape == (BATCH_SIZE,)
        assert torch.isfinite(loss).all()

    @pytest.mark.parametrize("num_classes", [2, 6])
    def test_log_prob_with_alternative_num_classes(
        self, denoiser_kwargs: dict, num_classes: int
    ) -> None:
        """Check the loss path doesn't carry hardcoded assumption of four classes."""
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

    def test_compute_full_vlb_accumulates_all_timesteps(
        self,
        monkeypatch: pytest.MonkeyPatch,
        denoiser_kwargs: dict,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Check compute_full_vlb sums every timestep's term and the prior."""
        x, z = batch
        denoiser_kwargs = {**denoiser_kwargs, "num_timesteps": SMALL_TIMESTEPS}
        diffusion = AptaDiffDiffusion(
            denoise_fn=AptaDiffDenoiser(**denoiser_kwargs),
            num_classes=NUM_CLASSES,
            num_timesteps=SMALL_TIMESTEPS,
        )

        monkeypatch.setattr(
            _model, "compute_vlb_loss", lambda **kwargs: torch.ones(BATCH_SIZE)
        )

        total = diffusion.compute_full_vlb(x, z)

        log_x0 = _index_to_log_onehot(x, NUM_CLASSES)
        expected = SMALL_TIMESTEPS + diffusion.kl_prior(log_x0)
        assert torch.allclose(total, expected, atol=1e-5)

    def test_log_prob_backward_populates_denoiser_gradients(
        self,
        diffusion: AptaDiffDiffusion,
        denoiser: AptaDiffDenoiser,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Check backward() on the training loss populates finite denoiser gradients."""
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

    def test_vb_all_backward_populates_gradients(
        self, denoiser_kwargs: dict, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Check gradients reach the denoiser through the checkpointed vb_all path."""
        x, z = batch
        denoiser_kwargs = {**denoiser_kwargs, "num_timesteps": SMALL_TIMESTEPS}
        small_denoiser = AptaDiffDenoiser(**denoiser_kwargs)
        diffusion = AptaDiffDiffusion(
            denoise_fn=small_denoiser,
            num_classes=NUM_CLASSES,
            num_timesteps=SMALL_TIMESTEPS,
            loss_type="vb_all",
        )
        diffusion.train()

        loss = diffusion.log_prob(x, z)
        loss.sum().backward()

        for param in small_denoiser.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    @pytest.mark.parametrize(
        "device_type, dtype",
        [
            ("cpu", torch.bfloat16),
            pytest.param(
                "cuda",
                torch.float16,
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_log_prob_finite_under_autocast(
        self,
        diffusion: AptaDiffDiffusion,
        denoiser: AptaDiffDenoiser,
        batch: tuple[torch.Tensor, torch.Tensor],
        device_type: str,
        dtype: torch.dtype,
    ) -> None:
        """Check loss and gradients stay finite under autocast."""
        x, z = batch
        diffusion = diffusion.to(device_type)
        x, z = x.to(device_type), z.to(device_type)
        diffusion.train()

        with torch.autocast(device_type=device_type, dtype=dtype):
            loss = diffusion.log_prob(x, z)

        assert torch.isfinite(loss).all()
        loss.sum().backward()
        for param in denoiser.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
