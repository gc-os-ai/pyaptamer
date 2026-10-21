"""Tests for the RaptGenLightning training wrapper"""

__author__ = ["siddharth7113"]

import pytest
import torch

from pyaptamer.raptgen import CNNPHMMVAE, RaptGenLightning


@pytest.fixture
def module():
    """RaptGenLightning around a small CNNPHMMVAE."""
    torch.manual_seed(0)
    return RaptGenLightning(CNNPHMMVAE(motif_len=5, hidden_size=8), force_epochs=2)


@pytest.mark.parametrize("step_method", ["training_step", "validation_step"])
def test_step_returns_scalar(module, step_method):
    """training_step and validation_step return a finite scalar tensor."""
    batch = (torch.randint(0, 4, (4, 12)),)
    loss = getattr(module, step_method)(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_beta_ramps_then_holds():
    """The KL weight rises linearly over beta_epochs and then stays at beta."""
    module = RaptGenLightning(
        CNNPHMMVAE(motif_len=5, hidden_size=8), beta=2.0, beta_epochs=4
    )
    assert module._beta_at(1) == pytest.approx(0.5)
    assert module._beta_at(2) == pytest.approx(1.0)
    assert module._beta_at(10) == 2.0


def test_beta_constant_without_schedule():
    """With beta_schedule=False the KL weight is beta from the first epoch."""
    module = RaptGenLightning(
        CNNPHMMVAE(motif_len=5, hidden_size=8), beta=0.5, beta_schedule=False
    )
    assert module._beta_at(1) == 0.5


def test_configure_optimizers(module):
    """configure_optimizers returns Adam with the configured learning rate."""
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == module.lr


def test_configure_optimizers_custom():
    """optimizer and optimizer_kwargs are forwarded."""
    module = RaptGenLightning(
        CNNPHMMVAE(motif_len=5, hidden_size=8),
        optimizer=torch.optim.SGD,
        optimizer_kwargs={"momentum": 0.9},
        lr=0.1,
    )
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_training_step_matches_loss_fn_with_force_matching():
    """At epoch 1 the step equals loss_fn with beta 1/beta_epochs and match_cost 5 - 4/force_epochs."""  # noqa: E501
    torch.manual_seed(0)
    model = CNNPHMMVAE(motif_len=5, hidden_size=8)
    module = RaptGenLightning(model, beta_epochs=4, force_epochs=2)
    data = torch.randint(0, 4, (4, 12))

    torch.manual_seed(1)
    loss = module.training_step((data,), batch_idx=0)
    torch.manual_seed(1)
    expected = model.loss_fn(
        data, *model(data), beta=0.25, force_matching=True, match_cost=3.0
    )
    assert torch.isclose(loss, expected)
