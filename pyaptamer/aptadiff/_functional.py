__author__ = ["aditi-dsi"]

import math

import torch


def log_one_minus_exp(a: torch.Tensor) -> torch.Tensor:
    """Computes log(1 - exp(a)) with numerical stability to avoid underflow."""
    return torch.log(-torch.expm1(a))


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes log(exp(a) + exp(b)) with numerical stability."""
    return torch.logaddexp(a, b)


def extract(schedule: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Gathers values from a 1D schedule array at specified time indices and
    reshapes them for broadcasting.
    """
    gathered = schedule.gather(-1, t)

    broadcast_shape = [t.shape[0]] + [1] * (len(x_shape) - 1)
    return gathered.view(broadcast_shape)


def cosine_alpha_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Generates a cosine-based noise schedule returning alpha values.

    This schedule determines the rate at which noise is added at each step
    of the forward diffusion process, based on the cosine schedule.

    Parameters
    ----------
    timesteps : int
        The total number of diffusion steps.
    s : float, optional
        A small offset value to prevent the noise at the very first step
        from being exactly zero, by default 0.008.

    Returns
    -------
    torch.Tensor
        A 1D tensor of square-rooted alpha values for each diffusion step,
        bounded to prevent numerical instability.

    References
    ----------
    .. [1] Nichol, A. Q., & Dhariwal, P. "Improved denoising diffusion
           probabilistic models." Proceedings of the 38th International Conference
           on Machine Learning, PMLR 139:8162-8171 (2021).
           https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)

    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.clip(alphas, min=0.001, max=1.0)
    alphas = torch.sqrt(alphas)

    return alphas.to(torch.float32)


def q_pred_one_timestep(
    log_x_prev: torch.Tensor, t: torch.Tensor, log_alpha: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the forward transition probability from step t-1 to t.

    Computes the probability distribution of the noisy sequence at the
    current step t, given the sequence state at the immediately preceding
    step t-1.

    Parameters
    ----------
    log_x_prev : torch.Tensor
        The log-encoded sequence state at the previous diffusion step (t-1).
    t : torch.Tensor
        A 1D tensor of time step indices for the current batch.
    log_alpha : torch.Tensor
        The 1D schedule array containing the log alpha (1 - beta) values.

    Returns
    -------
    torch.Tensor
        The log probability distribution of the sequence at step t.
    """
    log_alpha_t = extract(log_alpha, t, log_x_prev.shape)
    log_beta_t = log_one_minus_exp(log_alpha_t)

    log_probs = log_add_exp(log_x_prev + log_alpha_t, log_beta_t - math.log(4))

    return log_probs


def q_pred(
    log_x_0: torch.Tensor, t: torch.Tensor, log_alpha_bar: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the forward probability distribution going directly to step t.

    Computes the probability distribution of the noisy sequence at any
    arbitrary step t, starting directly from the clean data without
    iterating through intermediate steps.

    Parameters
    ----------
    log_x_0 : torch.Tensor
        The initial clean, log-encoded sequence data.
    t : torch.Tensor
        A 1D tensor of time step indices for the current batch.
    log_alpha_bar : torch.Tensor
        The 1D schedule array containing cumulative log alpha values.

    Returns
    -------
    torch.Tensor
        The log probability distribution of the noisy sequence at step t.
    """
    log_alpha_bar_t = extract(log_alpha_bar, t, log_x_0.shape)
    log_one_minus_alpha_bar_t = log_one_minus_exp(log_alpha_bar_t)

    log_probs = log_add_exp(
        log_x_0 + log_alpha_bar_t, log_one_minus_alpha_bar_t - math.log(4)
    )

    return log_probs
