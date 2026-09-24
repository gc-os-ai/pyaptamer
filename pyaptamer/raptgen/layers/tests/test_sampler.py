"""Tests for ProfileHMMSampler"""

__author__ = ["NoorMajdoub"]

import pytest
import torch

from pyaptamer.raptgen.layers._decoder import DecoderPHMM
from pyaptamer.raptgen.layers._sampler import ProfileHMMSampler


def _random_initialised_decoder_output(motif_len, embed_size):
    decoder = DecoderPHMM(
        motif_len=motif_len, embed_size=embed_size, hidden_size=embed_size
    )
    decoder.eval()
    x = torch.randn(1, embed_size)
    with torch.no_grad():
        transition_proba, emission_proba = decoder(x)
    return transition_proba[0].detach().numpy(), emission_proba[0].detach().numpy()


@pytest.mark.timeout(2)
@pytest.mark.parametrize("seed", range(20))
def test_most_probable_terminates(seed):
    motif_len = 10
    embed_size = 8

    torch.manual_seed(seed)
    a, e = _random_initialised_decoder_output(motif_len, embed_size)
    sampler = ProfileHMMSampler(a, e, proba_is_log=True)

    states, seq = sampler.most_probable()

    assert isinstance(seq, str)
    assert len(states) > 0
