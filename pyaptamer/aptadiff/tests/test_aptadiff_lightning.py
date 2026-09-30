"""Test the AptaDiffLightning training wrapper."""

__author__ = ["aditi-dsi"]

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from pyaptamer.aptadiff import AptaDiffDenoiser, AptaDiffDiffusion, AptaDiffLightning

NUM_CLASSES = 4
SEQ_LEN = 8
ENC_EMBED_SIZE = 16
TIMESTEPS = 4


@pytest.fixture
def diffusion() -> AptaDiffDiffusion:
    """Build a small AptaDiffDiffusion."""
    denoiser = AptaDiffDenoiser(
        enc_embed_size=ENC_EMBED_SIZE,
        num_classes=NUM_CLASSES,
        dim=32,
        depth=1,
        n_blocks=1,
        max_seq_len=SEQ_LEN,
        num_timesteps=TIMESTEPS,
        heads=2,
    )
    return AptaDiffDiffusion(
        denoise_fn=denoiser, num_classes=NUM_CLASSES, num_timesteps=TIMESTEPS
    )


class TestAptaDiffLightning:
    """Test AptaDiffLightning."""

    @pytest.fixture
    def model_lightning(self, diffusion: AptaDiffDiffusion) -> AptaDiffLightning:
        """Build an AptaDiffLightning with default parameters."""
        return AptaDiffLightning(diffusion)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("step_method", ["training_step", "validation_step"])
    def test_step(
        self, model_lightning: AptaDiffLightning, batch_size: int, step_method: str
    ) -> None:
        """Check training_step and validation_step return a non-negative scalar."""
        tokens = torch.randint(0, NUM_CLASSES, (batch_size, SEQ_LEN))
        x = F.one_hot(tokens, NUM_CLASSES).permute(0, 2, 1).float()
        z = torch.randn(batch_size, ENC_EMBED_SIZE)

        loss = getattr(model_lightning, step_method)((x, z), batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "optimizer_name, expected_cls",
        [
            ("adam", torch.optim.Adam),
            ("ADAM", torch.optim.Adam),
            ("sgd", torch.optim.SGD),
        ],
    )
    def test_optimizer_name_is_case_insensitive(
        self, optimizer_name: str, expected_cls: type[torch.optim.Optimizer]
    ) -> None:
        """Check optimizer_name selects the torch.optim class in any case."""
        optimizer_cls, kwargs = AptaDiffLightning.get_optimizer_cls_and_kwargs(
            optimizer_name, None, lr=1e-3
        )

        assert optimizer_cls is expected_cls
        assert kwargs == {"lr": 1e-3}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"optimizer_name": "adamm"},
            {"optimizer_name": "lr_scheduler"},
            {"optimizer_kwargs": {"lr": 0.1}},
            {"optimizer_name": "sgd", "optimizer_kwargs": {"betas": (0.9, 0.999)}},
            {"gamma": 0.0},
            {"gamma": 1.5},
        ],
    )
    def test_invalid_params_raise(
        self, diffusion: AptaDiffDiffusion, kwargs: dict[str, Any]
    ) -> None:
        """Check an invalid optimizer or gamma raises a ValueError."""
        with pytest.raises(ValueError):
            AptaDiffLightning(diffusion, **kwargs)

    def test_configure_optimizers_custom(self, diffusion: AptaDiffDiffusion) -> None:
        """Check custom optimizer settings and gamma are passed properly."""
        model_lightning = AptaDiffLightning(
            diffusion,
            optimizer_name="sgd",
            optimizer_kwargs={"momentum": 0.9},
            lr=1e-2,
            gamma=1.0,
        )

        config = model_lightning.configure_optimizers()
        optimizer = config["optimizer"]
        scheduler = config["lr_scheduler"]["scheduler"]

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 1e-2
        assert optimizer.defaults["momentum"] == 0.9
        assert scheduler.gamma == 1.0

    def test_configure_optimizers_uses_current_lr(
        self, model_lightning: AptaDiffLightning
    ) -> None:
        """Check changing a learning rate after construction takes effect."""
        model_lightning.lr = 0.5

        optimizer = model_lightning.configure_optimizers()["optimizer"]

        assert optimizer.defaults["lr"] == 0.5
