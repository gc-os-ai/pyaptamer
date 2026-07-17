
from enum import IntEnum, Enum
import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple

import logging

from pyaptamer.pyaptamer.raptgen.layers._utils import State, Transition
logger = logging.getLogger(__name__)


def kld_loss(mu, logvar):
    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) -
                            logvar.exp()) / mu.shape[0]
    return KLD


def ce_loss(recon_param, input):
    CE = F.cross_entropy(recon_param, input, reduction="sum") / input.shape[0]
    return CE
def profile_hmm_loss(recon_param, input, force_matching=False, match_cost=5):
    batch_size, random_len = input.shape

    a, e_m = recon_param
    motif_len = e_m.shape[1]

    F = torch.ones((batch_size, 3, motif_len + 1, random_len + 1),
                   device=input.device) * (-100)
    # init
    F[:, 0, 0, 0] = 0

    for i in range(random_len + 1):
        for j in range(motif_len + 1):
            # State M
            if j*i != 0:
                F[:, State.M, j, i] = e_m[:, j - 1].gather(1, input[:, i - 1:i])[:, 0] + \
                    torch.logsumexp(torch.stack((
                        a[:, j - 1, Transition.M2M] +
                        F[:, State.M, j - 1, i - 1],
                        a[:, j - 1, Transition.I2M] +
                        F[:, State.I, j - 1, i - 1],
                        a[:, j - 1, Transition.D2M] +
                        F[:, State.D, j - 1, i - 1])), dim=0)

            # State I
            if i != 0:
                F[:, State.I, j, i] = - 1.3863 + \
                    torch.logsumexp(torch.stack((
                        a[:, j, Transition.M2I] +
                        F[:, State.M, j, i-1],
                        # Removed D-to-I transition
                        # a[:, j, Transition.D2I] +
                        # F[:, State.D, j, i-1],
                        a[:, j, Transition.I2I] +
                        F[:, State.I, j, i-1]
                    )), dim=0)

            # State D
            if j != 0:
                F[:, State.D, j, i] = \
                    torch.logsumexp(torch.stack((
                        a[:, j - 1, Transition.M2D] +
                        F[:, State.M, j - 1, i],
                        # REMOVED I-to-D transition
                        # a[:, j - 1, Transition.I2D] +
                        # F[:, State.I, j - 1, i],
                        a[:, j - 1, Transition.D2D] +
                        F[:, State.D, j - 1, i]
                    )), dim=0)

    # final I->M transition
    F[:, State.M, motif_len, random_len] += a[:,
                                              motif_len, Transition.M2M]
    F[:, State.I, motif_len, random_len] += a[:,
                                              motif_len, Transition.I2M]
    F[:, State.D, motif_len, random_len] += a[:,
                                              motif_len, Transition.D2M]

    if force_matching:
        force_loss = np.log((match_cost+1)*match_cost/2) + \
            torch.sum((match_cost-1) * a[:, :, Transition.M2M], dim=1).mean()
        return - force_loss - torch.logsumexp(F[:, :, motif_len, random_len], dim=1).mean()
    return - torch.logsumexp(F[:, :, motif_len, random_len], dim=1).mean()


def profile_hmm_loss_fn_fast(input, recon_param, mu, logvar, debug=False, test=False, beta=1, force_matching=False, match_cost=5):
    phmmloss = torch_multi_polytope_dp_log(*recon_param, input, force_matching, match_cost)
    kld = kld_loss(mu, logvar)

    if debug:
        logger.info(f"phmm={phmmloss:.2f}, kld={kld:.2f}")
    if test:
        return phmmloss.item(), kld.item()
    return phmmloss + beta * kld



def torch_multi_polytope_dp_log(transition_proba, emission_proba, output, force_matching=False, match_cost=5):
    """
    torch_multi_polytope_dp_log(
        transition_proba,
        emission_proba,
        output
    )

    Given logarithmic parameters, the function calculates 
    the probability of the output sequence of a certain 
    Profile Hidden Markov Model (PHMM) by forward algorithm.

    For the efficiency, this function is utilizing polytope 
    model which enables parallel dynamic programming (DP).

    Parameters
    ----------
    transition_proba : torch.Tensor
        the tensor which define the probability to transit
        state to state. the tensor shape has to be 
        (`batch`, `from`=3, `to`=3, `model_length`+1) and
        the tensor has to be logarithmic number

    emission_proba : torch.Tensor
        the tensor which emit characters. The tensor shape 
        has to be: (`batch`, `model_length`, `augc`=4)

    output : torch.Tensor
        the tensor of the output vector. the tensor shape
        has to be (`batch`, `string_length`)

    Returns
    ------
    probabilities : torch.Tensor
        log-probabilities of the given output tensor with
        shape (`batch`,)

    """
    model_length = emission_proba.shape[1]
    batch_size, string_length = output.shape

    F = torch.ones(
        size=(batch_size, 3, model_length +
              string_length + 1, string_length + 1),
        device=output.device) * - 200
    F[:, State.M, 0, 0] = 0
    log4 = torch.Tensor([4]).log().to(output.device)
    arange = torch.arange(
        start=0, end=model_length + string_length + 1, device=output.device)
    for model_index_pre in range(1, model_length + string_length+1):
        if max(1, model_index_pre-model_length) < min(string_length+1, model_index_pre):
            m_slice = arange[max(1, model_index_pre-model_length)                             : min(string_length+1, model_index_pre)]
            F[:, State.M, model_index_pre, m_slice] = \
                torch.gather(
                    emission_proba[:, model_index_pre - m_slice - 1], 2,
                    output[:, m_slice - 1, None]).reshape(batch_size, len(m_slice)) \
                + torch.logsumexp(
                    transition_proba[:, :, State.M,
                                     model_index_pre - m_slice - 1]
                    + F[:, :, model_index_pre - 2, m_slice - 1], axis=1)

        if max(1, model_index_pre-model_length) < min(string_length+1, model_index_pre+1):
            i_slice = arange[max(1, model_index_pre-model_length)                             : min(string_length+1, model_index_pre+1)]
            F[:, State.I, model_index_pre, i_slice] = \
                torch.logsumexp(
                    transition_proba[:, :, State.I, model_index_pre - i_slice]
                    + F[:, :, model_index_pre - 1, i_slice - 1], axis=1) \
                - log4

        if max(0, model_index_pre-model_length) < min(string_length+1, model_index_pre):
            d_slice = arange[max(1, model_index_pre-model_length)                             : min(string_length+1, model_index_pre)]
            F[:, State.D, model_index_pre, d_slice] = \
                torch.logsumexp(
                    transition_proba[:, :, State.D,
                                     model_index_pre - d_slice - 1]
                    + F[:, :, model_index_pre - 1, d_slice], axis=1)
    if force_matching:
        return -torch.logsumexp(F[:, :, -1, -1] + transition_proba[:, :, State.M, -1], axis=1).mean()\
            - np.log((match_cost+1)*match_cost/2) \
            - torch.sum((match_cost-1) * transition_proba[:, State.M, State.M, :], dim=1).mean()
    return -torch.logsumexp(F[:, :, -1, -1] + transition_proba[:, :, State.M, -1], axis=1).mean()


def end_padded_multi_categorical_loss_fn(input, recon_param, mu, logvar, debug=False, test=False, beta=1):
    from pyaptamer.pyaptamer.raptgen.layers._utils import nt_index
    loss = multi_categorical_loss_fn(
        F.pad(input, (0, 1), "constant", nt_index.EOS),
        recon_param, mu, logvar, debug, test, beta)
    # logger.info(loss.shape)
    return loss


def multi_categorical_loss_fn(input, recon_param, mu, logvar, debug=False, test=False, beta=1):
    ce = ce_loss(recon_param, input)
    kld = kld_loss(mu, logvar)

    if debug:
        logger.info(f"ce={ce:.2f}, kld={kld:.2f}")
    if test:
        return ce.item(), kld.item()
    return ce + beta * kld


def profile_hmm_loss_fn(input, recon_param, mu, logvar, debug=False, test=False, beta=1, force_matching=False, match_cost=5):
    phmmloss = profile_hmm_loss(
        recon_param, input, force_matching=force_matching, match_cost=match_cost)
    kld = kld_loss(mu, logvar)

    if debug:
        logger.info(f"phmm={phmmloss:.2f}, kld={kld:.2f}")
    if test:
        return phmmloss.item(), kld.item()
    return phmmloss + beta * kld