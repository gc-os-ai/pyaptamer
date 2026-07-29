"""Test suite for AptaDiff's core diffusion functions."""

__author__ = ["aditi-dsi"]


import pytest
import torch

from pyaptamer.aptadiff._functional import (
    compute_vlb_loss,
    cosine_alpha_schedule,
    multinomial_kl,
    q_forward,
    q_forward_one_step,
    q_posterior,
)

B, K, L = 2, 4, 10  # standard shape: (batch, num_classes, seq_len)
TIMESTEPS = 1000


@pytest.fixture
def base_inputs():
    """Provides standard data and schedules for the tests."""
    log_x0 = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    alphas = cosine_alpha_schedule(TIMESTEPS)

    log_alpha = torch.log(alphas)
    log_alphabar = torch.cumsum(log_alpha, dim=0)

    return log_x0, log_alpha, log_alphabar


@pytest.mark.parametrize("t_val", [0, 500, 999])
def test_forward_processes_are_valid_distributions(base_inputs, t_val):
    """Verify forward step distributions sum to 1 over nucleotide classes."""
    log_x0, log_alpha, log_alphabar = base_inputs

    t = torch.full((B,), t_val, dtype=torch.long)

    out_one_step = q_forward_one_step(log_x0, t, log_alpha, num_classes=K)
    out_direct = q_forward(log_x0, t, log_alphabar, num_classes=K)

    prob_one_step = torch.exp(out_one_step)
    prob_direct = torch.exp(out_direct)

    expected_ones = torch.ones((B, L))

    assert torch.allclose(torch.sum(prob_one_step, dim=1), expected_ones, atol=1e-5)
    assert torch.allclose(torch.sum(prob_direct, dim=1), expected_ones, atol=1e-5)


@pytest.mark.parametrize("t_val", [1, 500, 999])
def test_posterior_is_valid_distribution(base_inputs, t_val):
    """Verify the reverse step distribution sums to 1 over nucleotide classes."""
    log_x0, log_alpha, log_alphabar = base_inputs

    log_xt = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    t = torch.full((B,), t_val, dtype=torch.long)

    log_q = q_posterior(log_x0, log_xt, t, log_alpha, log_alphabar)
    prob_q = torch.exp(log_q)

    expected_ones = torch.ones((B, L))
    assert torch.allclose(torch.sum(prob_q, dim=1), expected_ones, atol=1e-5)


def test_posterior_t0_returns_x0_exactly(base_inputs):
    """Verify q_posterior returns log_x0 exactly when t=0."""
    log_x0, log_alpha, log_alphabar = base_inputs
    log_xt = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    t = torch.zeros((B,), dtype=torch.long)

    log_q = q_posterior(log_x0, log_xt, t, log_alpha, log_alphabar)

    assert torch.equal(log_q, log_x0)


@pytest.mark.parametrize("t_val", [0, 500])
def test_vlb_loss_piecewise_masking(base_inputs, t_val):
    """Ensure VLB loss uses cross-entropy at t=0 and KL divergence for t>0."""
    log_x0, log_alpha, log_alphabar = base_inputs

    log_pred = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    log_xt = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    t = torch.full((B,), t_val, dtype=torch.long)

    log_q = q_posterior(log_x0, log_xt, t, log_alpha, log_alphabar)
    loss = compute_vlb_loss(log_x0, log_xt, t, log_alpha, log_alphabar, log_pred)

    if t_val == 0:
        expected_nll = -torch.sum(torch.exp(log_x0) * log_pred, dim=[1, 2])
        assert torch.allclose(loss, expected_nll, atol=1e-5)
    else:
        expected_kld = torch.sum(multinomial_kl(log_q, log_pred), dim=1)
        assert torch.allclose(loss, expected_kld, atol=1e-5)
