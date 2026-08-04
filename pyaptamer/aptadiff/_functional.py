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
    Extracts specific schedule values (alpha) from a 1D schedule array
    based on batch time indices.

    Parameters
    ----------
    schedule : torch.Tensor
        A 1D tensor containing the full schedule (e.g., all alpha values
        for 1000 steps).
    t : torch.Tensor
        A 1D tensor of time step indices for the current batch.
    x_shape : tuple
        The shape of the target data tensor (e.g., (batch_size, num_classes, seq_len)).

    Returns
    -------
    torch.Tensor
        The extracted schedule values, reshaped to broadcast against the target data.
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

    alpha_bar = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]

    alphas = alpha_bar[1:] / alpha_bar[:-1]
    alphas = torch.clip(alphas, min=0.001, max=1.0)
    alphas = torch.sqrt(alphas)

    return alphas.to(torch.float32)


def q_forward_one_step(
    log_x_prev: torch.Tensor,
    t: torch.Tensor,
    log_alpha: torch.Tensor,
    num_classes: int = 4,
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
        The 1D schedule array containing the log alpha values.
    num_classes : int
        The number of unique nucleotides in the sequence. Default is 4.

    Returns
    -------
    torch.Tensor
        The log probability distribution of the sequence at step t.
    """
    log_alpha_t = extract(log_alpha, t, log_x_prev.shape)
    log_beta_t = log_one_minus_exp(log_alpha_t)

    log_proba = log_add_exp(
        log_x_prev + log_alpha_t, log_beta_t - math.log(num_classes)
    )

    return log_proba


def q_forward(
    log_x0: torch.Tensor,
    t: torch.Tensor,
    log_alphabar: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Calculates the forward probability distribution going directly to step t.

    Computes the probability distribution of the noisy sequence at any
    arbitrary step t, starting directly from the clean data (x_0) without
    iterating through intermediate steps.

    Parameters
    ----------
    log_x0 : torch.Tensor
        The initial clean, log-encoded sequence data.
    t : torch.Tensor
        A 1D tensor of time step indices for the current batch.
    log_alphabar : torch.Tensor
        The 1D schedule array containing cumulative log alpha values.
    num_classes : int
        The number of unique nucleotides in the sequence. Default is 4.

    Returns
    -------
    torch.Tensor
        The log probability distribution of the noisy sequence at step t.
    """
    log_alphabar_t = extract(log_alphabar, t, log_x0.shape)
    log_1m_alphabar_t = log_one_minus_exp(log_alphabar_t)

    log_proba = log_add_exp(
        log_x0 + log_alphabar_t, log_1m_alphabar_t - math.log(num_classes)
    )

    return log_proba


def q_posterior(
    log_x0: torch.Tensor,
    log_xt: torch.Tensor,
    t: torch.Tensor,
    log_alpha: torch.Tensor,
    log_alphabar: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the reverse step probability distribution x_{t-1} given
    x_t and x_0 (or q(x_{t-1} | x_t, x_0)).

    Parameters
    ----------
    log_x0 : torch.Tensor
        The initial clean, log-encoded sequence data.
    log_xt : torch.Tensor
        The log encoded noisy sequence at a current step t.
    t : torch.Tensor
        A 1D tensor of time step indices for the current batch.
    log_alpha : torch.Tensor
        The 1D schedule array containing the log alpha values.
    log_alphabar : torch.Tensor
        The 1D schedule array containing cumulative log alpha values.

    Returns
    -------
    torch.Tensor
        Normalized log posterior for x_{t-1}. For t == 0, returns log_x0.
    """
    t_prev = torch.clamp(t - 1, min=0)

    log_prior = q_forward(log_x0, t_prev, log_alphabar)
    log_likelihood = q_forward_one_step(log_xt, t, log_alpha)
    log_posterior = log_prior + log_likelihood
    normalized_log_posterior = log_posterior - torch.logsumexp(
        log_posterior, dim=1, keepdim=True
    )

    t_broadcast = t.view(-1, 1, 1)

    return torch.where(t_broadcast == 0, log_x0, normalized_log_posterior)


def multinomial_kl(log_q: torch.Tensor, log_pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence between the target and predicted categorical
    distributions at each sequence position.

    Parameters
    ----------
    log_q : torch.Tensor
        The log probabilities of the true posterior distribution
        for the reverse step.
    log_pred : torch.Tensor
        The log probability distribution predicted by the neural network.

    Returns
    -------
    torch.Tensor
        The KL divergence loss between the true and predicted probability
        distributions for each nucleotide's position in the sequence.
    """
    q = torch.exp(log_q)
    kl_divergence = torch.sum(q * (log_q - log_pred), dim=1)

    return kl_divergence


def compute_vlb_loss(
    log_x0: torch.Tensor,
    log_xt: torch.Tensor,
    t: torch.Tensor,
    log_alpha: torch.Tensor,
    log_alphabar: torch.Tensor,
    log_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the Variational Lower Bound (VLB) loss for a specific time step,
    mathematically denoted as L_t.

    This function computes the loss for a single randomly sampled time step per
    sequence in the batch.

    Because the mathematical definition of L_t is piecewise, the formula depends
    on the specific time step sampled for each sequence:
    - If the sampled step is t > 0: Computes the KL divergence between the true
      denoising posterior and the neural network's prediction.
    - If the sampled step is t = 0: Computes a standard cross-entropy
      reconstruction loss (Negative Log-Likelihood) against the clean sequence.

    Parameters
    ----------
    log_x0 : torch.Tensor
        The initial clean, log-encoded sequence data.
    log_xt : torch.Tensor
        The log encoded noisy sequence at a current step t.
    t : torch.Tensor
        A 1D tensor of time step indices for the current batch.
    log_alpha : torch.Tensor
        The 1D schedule array containing the log alpha values.
    log_alphabar : torch.Tensor
        The 1D schedule array containing cumulative log alpha values.
    log_pred : torch.Tensor
        The log probability distribution predicted by the neural network.

    Returns
    -------
    torch.Tensor
        A 1D tensor containing the computed loss value for each sample in the batch.
    """
    log_q = q_posterior(log_x0, log_xt, t, log_alpha, log_alphabar)
    kld = torch.sum(multinomial_kl(log_q, log_pred), dim=1)

    decoder_nll = -torch.sum(torch.exp(log_x0) * log_pred, dim=[1, 2])

    mask = (t == 0).float()
    loss = mask * decoder_nll + (1.0 - mask) * kld

    return loss
