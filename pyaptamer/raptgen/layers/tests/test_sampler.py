"""Tests for ProfileHMMSampler"""

__author__ = ["NoorMajdoub"]

import numpy as np
import pytest
import torch

from pyaptamer.raptgen.layers._decoder import DecoderPHMM
from pyaptamer.raptgen.layers._loss import profile_hmm_loss
from pyaptamer.raptgen.layers._sampler import ProfileHMMSampler
from pyaptamer.raptgen.layers._utils import State, Transition, seq_to_indices


def _random_initialised_decoder_output(motif_len, embed_size):
    decoder = DecoderPHMM(
        motif_len=motif_len, embed_size=embed_size, hidden_size=embed_size
    )
    decoder.eval()
    x = torch.randn(1, embed_size)
    with torch.no_grad():
        transition_proba, emission_proba = decoder(x)
    return transition_proba[0].detach().numpy(), emission_proba[0].detach().numpy()


@pytest.mark.parametrize("seed", range(20))
def test_most_probable_terminates(seed):
    """Greedy walk on random decoder output returns a string and a state list."""
    motif_len = 10
    embed_size = 8

    torch.manual_seed(seed)
    a, e = _random_initialised_decoder_output(motif_len, embed_size)
    sampler = ProfileHMMSampler(a, e, proba_is_log=True)

    states, seq = sampler.most_probable()

    assert isinstance(seq, str)
    assert len(states) > 0


def test_most_probable_insert_at_every_position():
    """Walk M I M I M on a one-position model finishes within the step bound."""
    a = np.zeros((2, 7))
    a[:, Transition.M2I] = 1.0
    e = np.array([[0.7, 0.1, 0.1, 0.1]])
    sampler = ProfileHMMSampler(a, e)

    states, seq = sampler.most_probable()

    assert seq == "NAN"
    assert [s for _, s in states] == [State.M, State.I, State.M, State.I, State.M]


def test_calc_seq_proba_matches_profile_hmm_loss():
    """calc_seq_proba equals the negated batched forward-algorithm loss."""
    torch.manual_seed(0)
    motif_len, embed_size = 6, 8
    decoder = DecoderPHMM(
        motif_len=motif_len, embed_size=embed_size, hidden_size=embed_size
    ).eval()
    with torch.no_grad():
        a, e = decoder(torch.randn(1, embed_size))
    sampler = ProfileHMMSampler(a[0].numpy(), e[0].numpy(), proba_is_log=True)

    seq = "ATGCATGC"
    loss = profile_hmm_loss((a, e), torch.tensor([seq_to_indices(seq)]))

    assert torch.allclose(sampler.calc_seq_proba(seq), -loss, atol=1e-4)
