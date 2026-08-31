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


@pytest.fixture
def clamped_one_hot_inputs():
    """Provides clamped one-hot log_x0 data instead of uniform random probabilities."""
    labels = torch.randint(0, K, (B, L))
    probs = torch.full((B, K, L), 1e-30)
    probs.scatter_(1, labels.unsqueeze(1), 1.0 - (K - 1) * 1e-30)

    log_x0 = torch.log(probs)
    alphas = cosine_alpha_schedule(TIMESTEPS)

    log_alpha = torch.log(alphas)
    log_alphabar = torch.cumsum(log_alpha, dim=0)

    return log_x0, log_alpha, log_alphabar


@pytest.mark.parametrize("t_val", [0, 500, 999])
def test_forward_processes_are_valid_distributions(base_inputs, t_val):
    """Verify forward step distributions sum to 1 over nucleotide classes."""
    log_x0, log_alpha, log_alphabar = base_inputs

    t = torch.full((B,), t_val, dtype=torch.long)

    out_one_step = q_forward_one_step(log_x0, t, log_alpha)
    out_direct = q_forward(log_x0, t, log_alphabar)

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


def test_posterior_t0_uses_likelihood(base_inputs):
    """Verify q_posterior at t=0 depends on log_xt via the likelihood term."""
    log_x0, log_alpha, log_alphabar = base_inputs
    t = torch.zeros((B,), dtype=torch.long)

    log_xt_a = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    log_xt_b = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))

    log_q_a = q_posterior(log_x0, log_xt_a, t, log_alpha, log_alphabar)
    log_q_b = q_posterior(log_x0, log_xt_b, t, log_alpha, log_alphabar)

    assert not torch.allclose(log_q_a, log_q_b)


def test_posterior_t0_reference_formula(base_inputs):
    """Verify q_posterior at t=0 matches the reference prior-likelihood formula."""
    log_x0, log_alpha, log_alphabar = base_inputs
    log_xt = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    t = torch.zeros((B,), dtype=torch.long)

    log_q = q_posterior(log_x0, log_xt, t, log_alpha, log_alphabar)

    log_likelihood = q_forward_one_step(log_xt, t, log_alpha)
    unnormalized = log_x0 + log_likelihood
    expected = unnormalized - torch.logsumexp(unnormalized, dim=1, keepdim=True)

    assert torch.allclose(log_q, expected, atol=1e-5)


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


def test_posterior_t0_clamped_inputs(clamped_one_hot_inputs):
    """Verify q_posterior at t=0 stays valid and keeps the clamped class dominant."""
    log_x0, log_alpha, log_alphabar = clamped_one_hot_inputs
    log_xt = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    t = torch.zeros((B,), dtype=torch.long)

    log_q = q_posterior(log_x0, log_xt, t, log_alpha, log_alphabar)
    prob_q = torch.exp(log_q)

    expected_ones = torch.ones((B, L))
    assert torch.allclose(torch.sum(prob_q, dim=1), expected_ones, atol=1e-5)

    dominant_prob = prob_q.gather(1, log_x0.argmax(dim=1, keepdim=True))
    assert torch.all(dominant_prob > 0.99)


def test_vlb_loss_clamped_inputs(clamped_one_hot_inputs):
    """Verify VLB loss stays finite and matches the decoder NLL for clamped inputs."""
    log_x0, log_alpha, log_alphabar = clamped_one_hot_inputs
    log_pred = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    log_xt = torch.log(torch.nn.functional.normalize(torch.rand(B, K, L), p=1, dim=1))
    t = torch.zeros((B,), dtype=torch.long)

    loss = compute_vlb_loss(log_x0, log_xt, t, log_alpha, log_alphabar, log_pred)

    assert torch.isfinite(loss).all()
    expected_nll = -torch.sum(torch.exp(log_x0) * log_pred, dim=[1, 2])
    assert torch.allclose(loss, expected_nll, atol=1e-5)


def test_multinomial_kl_known_value():
    """Verify multinomial_kl against a pre-computed KL divergence value."""
    q_probs = torch.tensor([0.9, 0.1])
    pred_probs = torch.tensor([0.6, 0.4])

    log_q = torch.log(q_probs).view(1, 2, 1)
    log_pred = torch.log(pred_probs).view(1, 2, 1)

    kl = multinomial_kl(log_q, log_pred)

    expected = 0.22628916118535888  # KL(q=[0.9, 0.1] || pred=[0.6, 0.4])
    assert torch.allclose(kl.squeeze(), torch.tensor(expected), atol=1e-5)


def test_multinomial_kl_identical_inputs(base_inputs):
    """KL divergence between identical distributions must be zero."""
    log_x0, _, _ = base_inputs

    kl = multinomial_kl(log_x0, log_x0)

    assert torch.allclose(kl, torch.zeros(B, L), atol=1e-5)
